// engine_ext/src/poly/mixtral.hpp - Mixtral with routed experts (HF MixtralForCausalLM), as 8x7B is.
//
// StrataShift's file, not PolyStrata's: it is copied into a copy of the pinned PolyStrata tree (MIT, Copyright (c)
// 2026 Niko1221 and the Strata contributors; engine_ext/NOTICE.md) and built as strata-shift-poly.  It builds on
// the engine's Engine class and kernels, and its layers follow the engine's own qwen35.cpp.
//
// One residual stream and attention over every token so far in every layer, with a rotary embedding and no norm
// on the queries and keys.  A layer's FFN is routed SiLU experts alone: no shared expert and no dense MLP.  The
// rotary pairs are ggml's normal ones (value i with i + 1): StrataShift's emit orders the rows of the query and key
// matrices for that, as llama.cpp's converter does.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

class Mixtral final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void save_layer(int, int64_t, uint8_t*) override {}   // no recurrent state, no window
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return n_head_ * head_dim_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;

private:
    /// A layer's attention weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        Dev wq, wk, wv, wo;
        uint16_t* keys = nullptr;   ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
    };

    void attn(Block& B, int64_t pos, int n);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0;
    int n_attn_ = 0;              ///< the layers this engine runs, each with its caches
    Rope rope_;                   ///< plain rotation: no YaRN (ext_factor 0)
    int chunk_ = 1;               ///< tokens the attention takes at a time

    // device scratch
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *score_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
};

}  // namespace strata::poly
