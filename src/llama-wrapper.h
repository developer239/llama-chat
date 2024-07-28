// llama-wrapper.h
#pragma once

#include "llama.h"

#include <string>
#include <memory>
#include <functional>
#include <vector>

struct ModelParams {
  int n_gpu_layers = 0;
  bool vocab_only = false;
  bool use_mmap = true;
  bool use_mlock = false;
};

struct ContextParams {
  size_t n_ctx = 4096;
  int n_threads = 6;
  int n_batch = 512;
  bool logits_all = false;
  bool embedding = false;
};

struct SamplingParams {
  size_t max_tokens = 1000;
  float temperature = 0.8f;
  int32_t top_k = 45;
  float top_p = 0.95f;
  float repeat_penalty = 1.1f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  std::vector<llama_token> repeat_penalty_tokens;
};

class LlamaWrapper {
 public:
  LlamaWrapper();
  ~LlamaWrapper();

  LlamaWrapper(const LlamaWrapper&) = delete;
  LlamaWrapper& operator=(const LlamaWrapper&) = delete;

  LlamaWrapper(LlamaWrapper&&) noexcept = default;
  LlamaWrapper& operator=(LlamaWrapper&&) noexcept = default;

  // Model Initialization
  bool InitializeModel(const std::string& model_path, const ModelParams& params);

  // Context Initialization
  bool InitializeContext(const ContextParams& params);

  // Tokenization and Encoding
  std::vector<llama_token> Encode(const std::string& text, bool add_bos = true) const;
  std::string Decode(const std::vector<llama_token>& tokens) const;
  llama_token TokenBos() const;
  llama_token TokenEos() const;
  llama_token TokenNl() const;

  // Evaluation and Sampling
  std::string RunQuery(const std::string& prompt, const SamplingParams& params, bool add_bos = true) const;
  void RunQueryStream(const std::string& prompt, const SamplingParams& params, const std::function<void(const std::string&)>& callback, bool add_bos = true) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};
