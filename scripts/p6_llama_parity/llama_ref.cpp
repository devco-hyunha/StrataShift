// scripts/p6_llama_parity/llama_ref.cpp (p6-01 AC4): the pinned llama.cpp, CPU only, on a GGUF; after the prompt and
// after every greedy pick it writes the whole next-token distribution.
//   llama_ref MODEL.gguf TOKENS(1,2,...) N_NEW OUT_PREFIX
//   OUT_PREFIX.tokens: the greedy picks (comma separated); OUT_PREFIX.probs: n_vocab f32 probabilities a step
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: llama_ref MODEL TOKENS N_NEW OUT\n"); return 2; }
    std::vector<llama_token> toks;
    { std::stringstream ss(argv[2]); std::string s; while (std::getline(ss, s, ',')) toks.push_back(std::atoi(s.c_str())); }
    const int n_new = std::atoi(argv[3]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 1024;
    cp.n_threads = cp.n_threads_batch = 8;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE* fp = std::fopen((std::string(argv[4]) + ".probs").c_str(), "wb");
    std::string picked;
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    std::vector<float> p(n_vocab);
    llama_token next = 0;
    for (int step = 0; step <= n_new; ++step) {
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed at %d\n", step); return 1; }
        const float* lg = llama_get_logits_ith(ctx, -1);
        float mx = lg[0];
        for (int i = 1; i < n_vocab; ++i) mx = std::max(mx, lg[i]);
        double sum = 0;
        for (int i = 0; i < n_vocab; ++i) sum += p[i] = std::exp(lg[i] - mx);
        int best = 0;
        for (int i = 0; i < n_vocab; ++i) { p[i] = (float) (p[i] / sum); if (p[i] > p[best]) best = i; }
        std::fwrite(p.data(), sizeof(float), n_vocab, fp);
        if (step == n_new) break;
        next = best;
        picked += (picked.empty() ? "" : ",") + std::to_string(best);
        b = llama_batch_get_one(&next, 1);
    }
    std::fclose(fp);
    FILE* ft = std::fopen((std::string(argv[4]) + ".tokens").c_str(), "w");
    std::fprintf(ft, "%s\n", picked.c_str());
    std::fclose(ft);
    std::printf("n_vocab %d, picked %s\n", n_vocab, picked.c_str());
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
