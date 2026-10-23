// Run llama.cpp on a model one token at a time and dump every contiguous
// graph tensor of each decode step to OUTDIR/stepN/<idx>_<name>.bin plus a
// MANIFEST.tsv (idx, name, op, type, ne0..ne3, src names).
//
// This is the "CPU oracle" side: it is llama.cpp itself at the PS4 baseline
// commit, built with Jaguar-equivalent flags, not a reimplementation.
//
// usage: dump_layer0 MODEL.gguf OUTDIR FA(0=off|1=on|2=auto) THREADS TOKEN0 [TOKEN1 ...]
// Set DUMP_STOP_AFTER=l_out-0 to stop dumping each step after layer 0's output
// (keeps a 28-layer real-model dump to ~2 MB per step instead of ~60 MB).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

struct dump_ctx {
    std::string dir;
    FILE * manifest = nullptr;
    int idx = 0;
    std::string stop_after;
    bool done = false;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (dump_ctx *) ud;
    if (ask) {
        return true;
    }
    if (d->done || !ggml_is_contiguous(t)) {
        return true;
    }
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());

    char fname[512];
    std::string name = t->name;
    for (auto & c : name) {
        if (c == '/' || c == ' ' || c == '(' || c == ')') c = '_';
    }
    snprintf(fname, sizeof(fname), "%04d_%s.bin", d->idx, name.c_str());
    std::string path = d->dir + "/" + fname;
    FILE * f = fopen(path.c_str(), "wb");
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);

    fprintf(d->manifest, "%d\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld", d->idx, t->name, ggml_op_desc(t),
            ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
            (long long) t->ne[3]);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i]) fprintf(d->manifest, "\t%s", t->src[i]->name);
    }
    fprintf(d->manifest, "\n");
    d->idx++;
    if (!d->stop_after.empty() && d->stop_after == t->name) {
        d->done = true;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL OUTDIR FA THREADS TOKEN0 [TOKEN1 ...]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    std::string outdir = argv[2];
    const int fa = atoi(argv[3]);
    const int nthreads = atoi(argv[4]);
    std::vector<llama_token> toks;
    for (int i = 5; i < argc; i++) toks.push_back(atoi(argv[i]));

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;  // no weight repacking
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 2;

    dump_ctx d;
    if (const char * e = getenv("DUMP_STOP_AFTER")) d.stop_after = e;
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 32;
    cp.n_ubatch = 32;
    cp.n_threads = nthreads;
    cp.n_threads_batch = nthreads;
    cp.flash_attn_type = fa == 2 ? LLAMA_FLASH_ATTN_TYPE_AUTO : fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &d;
    cp.embeddings = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;

    mkdir(outdir.c_str(), 0755);
    for (size_t s = 0; s < toks.size(); s++) {
        d.dir = outdir + "/step" + std::to_string(s);
        mkdir(d.dir.c_str(), 0755);
        d.manifest = fopen((d.dir + "/MANIFEST.tsv").c_str(), "w");
        d.idx = 0;
        d.done = false;
        llama_token t = toks[s];
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "decode failed at step %zu\n", s);
            return 4;
        }
        fclose(d.manifest);
        fprintf(stderr, "step %zu: dumped %d tensors\n", s, d.idx);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
