#include "llama-wrapper.h"
#include "common.h"

class LlamaWrapper::Impl {
 public:
  Impl() {
    llama_backend_init();
  }

  ~Impl() {
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();
  }

  bool InitializeModel(const std::string& model_path, const ModelParams& params) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.vocab_only = params.vocab_only;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;

    model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
      throw std::runtime_error("Failed to load model from " + model_path);
    }
    return true;
  }

  bool InitializeContext(const ContextParams& params) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_batch = params.n_batch;
    ctx_params.logits_all = params.logits_all;
    ctx_params.embeddings = params.embedding;

    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
      throw std::runtime_error("Failed to create the llama_context");
    }
    return true;
  }

  std::vector<llama_token> Encode(const std::string& text, bool add_bos) const {
    return llama_tokenize(ctx, text, add_bos);
  }

  std::string Decode(const std::vector<llama_token>& tokens) const {
    std::string result;
    for (const auto& token : tokens) {
      result += llama_token_to_piece(ctx, token);
    }
    return result;
  }

  llama_token TokenBos() const { return llama_token_bos(model); }
  llama_token TokenEos() const { return llama_token_eos(model); }
  llama_token TokenNl() const { return llama_token_nl(model); }

  std::string RunQuery(const std::string& prompt, const SamplingParams& params, bool add_bos = true) const {
    std::vector<llama_token> tokens = Encode(prompt, add_bos);
    std::string result;

    llama_batch batch = llama_batch_init(params.max_tokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }

    if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t n_cur = batch.n_tokens;
    while (n_cur < params.max_tokens) {
      llama_token new_token_id = SampleToken(params);
      if (llama_token_is_eog(model, new_token_id)) break;

      result += llama_token_to_piece(ctx, new_token_id);
      llama_batch_clear(batch);
      llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
      n_cur += 1;

      if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
    return result;
  }

  void RunQueryStream(const std::string& prompt, const SamplingParams& params, const std::function<void(const std::string&)>& callback, bool add_bos = true) const {
    std::vector<llama_token> tokens = Encode(prompt, add_bos);

    llama_batch batch = llama_batch_init(params.max_tokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }

    if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t n_cur = batch.n_tokens;
    while (n_cur < params.max_tokens) {
      llama_token new_token_id = SampleToken(params);
      if (llama_token_is_eog(model, new_token_id)) break;

      std::string piece = llama_token_to_piece(ctx, new_token_id);
      callback(piece);

      llama_batch_clear(batch);
      llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
      n_cur += 1;

      if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
  }

 private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;

  llama_token SampleToken(const SamplingParams& params) const {
    auto logits = llama_get_logits(ctx);
    auto n_vocab = llama_n_vocab(model);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    if (!params.repeat_penalty_tokens.empty()) {
      llama_sample_repetition_penalties(
          ctx,
          &candidates_p,
          params.repeat_penalty_tokens.data(),
          params.repeat_penalty_tokens.size(),
          params.repeat_penalty,
          params.frequency_penalty,
          params.presence_penalty
      );
    }

    llama_sample_top_k(ctx, &candidates_p, params.top_k, 1);
    llama_sample_top_p(ctx, &candidates_p, params.top_p, 1);
    llama_sample_temp(ctx, &candidates_p, params.temperature);

    return llama_sample_token(ctx, &candidates_p);
  }
};

// Implement LlamaWrapper methods
LlamaWrapper::LlamaWrapper() : pimpl(std::make_unique<Impl>()) {}
LlamaWrapper::~LlamaWrapper() = default;

bool LlamaWrapper::InitializeModel(const std::string& model_path, const ModelParams& params) {
  return pimpl->InitializeModel(model_path, params);
}

bool LlamaWrapper::InitializeContext(const ContextParams& params) {
  return pimpl->InitializeContext(params);
}

std::vector<llama_token> LlamaWrapper::Encode(const std::string& text, bool add_bos) const {
  return pimpl->Encode(text, add_bos);
}

std::string LlamaWrapper::Decode(const std::vector<llama_token>& tokens) const {
  return pimpl->Decode(tokens);
}

llama_token LlamaWrapper::TokenBos() const { return pimpl->TokenBos(); }
llama_token LlamaWrapper::TokenEos() const { return pimpl->TokenEos(); }
llama_token LlamaWrapper::TokenNl() const { return pimpl->TokenNl(); }

std::string LlamaWrapper::RunQuery(const std::string& prompt, const SamplingParams& params, bool add_bos) const {
  return pimpl->RunQuery(prompt, params, add_bos);
}

void LlamaWrapper::RunQueryStream(const std::string& prompt, const SamplingParams& params, const std::function<void(const std::string&)>& callback, bool add_bos) const {
  pimpl->RunQueryStream(prompt, params, callback, add_bos);
}
