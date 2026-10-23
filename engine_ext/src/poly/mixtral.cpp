// engine_ext/src/poly/mixtral.cpp - Mixtral's layers (StrataShift's file; engine_ext/NOTICE.md).  The structure is
// the engine's qwen35.cpp attention layer without its gate and norms; the reference for the math is HF
// MixtralForCausalLM and llama.cpp's llama graph.
#include "mixtral.hpp"

#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum (qwen35.cpp)
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time (qwen35.cpp)

}  // namespace

void Mixtral::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_attn_ = 0;
}

void Mixtral::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : head_dim_;
    const MetaValue* window = find("attention.sliding_window");

    need(n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count");
    need(head_dim_ > 0 && head_dim_ % 2 == 0 && head_dim_ <= 2048 && shape_i("attention.value_length") == head_dim_,
         "the attention's keys and values differ in length, or are not an even width up to 2048");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");
    need(!window || window->u == 0, "a sliding window, which this architecture does not run");
    rope_.theta_scale = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);

    // the experts are SiLU with no limit on their inputs (gelu_ stays false)
    clamp_exp_.assign((size_t) n_layer_, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Mixtral::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a) { return upload(tensor(il, n), blk(il, n), a, 1, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    load_moe(il, false);   // no shared expert
    ++n_attn_;
}

// Every layer's caches: a row a position.
void Mixtral::alloc_caches() {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        B.keys = (uint16_t*) kv_alloc(row, 1);
        B.vals = (uint16_t*) kv_alloc(row, 1);
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Mixtral::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    q_ = fbuf(heads);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

size_t Mixtral::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void Mixtral::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void Mixtral::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// The recordings of a layer's head are numbered by the rows they are sized for.
int Mixtral::head_key(int, int n) const { return room_key(n); }

// Queries and keys turned by the position, without a norm; values as they are.  xn_ and xq_ hold the layer's
// input.  A recorded launch reads the rotary position from at_ (its second number a token, turn()).
void Mixtral::attn(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    const long long* turn_at = at ? at + 1 : nullptr;
    const int width = n_kv_ * head_dim_;
    lin(B.wq, xn_, q_, n, xq_);
    dsv4_rope(q_, q_, head_dim_, n_head_, 0, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_, turn_at);
    lin(B.wk, xn_, k_, n, xq_);
    dsv4_rope(k_, k_, head_dim_, n_kv_, 0, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_, turn_at);
    lin(B.wv, xn_, v_, n, xq_);
    if (at) {
        store_f16_at(k_, B.keys, width, at, n, 0, st_);
        store_f16_at(v_, B.vals, width, at, n, 0, st_);
    } else {
        store_f16(k_, B.keys, width, pos, n, st_);
        store_f16(v_, B.vals, width, pos, n, st_);
    }
    const float scale = 1.f / std::sqrt((float) head_dim_);
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false,
               at, at ? room(n) : 0);
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its router's copy.  A prompt batch's MoE (moe_rows) does its own routing.
void Mixtral::layer_head(Layer& L, int il, int n) {
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(blocks_[(size_t) il], pos_, n);
    dbg("attn", il, ao_, n_embd_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);

    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// No expert is shared: the routed experts' sum starts from nothing.
void Mixtral::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

void Mixtral::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Mixtral::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// HF Mixtral's router: a softmax over every expert's score, the n_used most likely, and their weights made to sum
// to one.  The same as the engine's Qwen35::route.
void Mixtral::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] /= denom;
}

bool Mixtral::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, true, 0.f, ids, w, st_);
    return true;
}

}  // namespace strata::poly
