#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

typedef int llama_token;

struct LlamaToken {
  llama_token tokenId;
  explicit LlamaToken(llama_token id = 0) : tokenId(id) {}
};

struct ModelParams {
  int nGpuLayers = 0;
  bool vocabularyOnly = false;
  bool useMemoryMapping = true;
  bool useModelLock = false;
};

struct ContextParams {
  size_t nContext = 4096;
  int nThreads = 6;
  int nBatch = 512;
};

struct SamplingParams {
  size_t maxTokens = 1000;
  float temperature = 0.8f;
  int32_t topK = 45;
  float topP = 0.95f;
  float repeatPenalty = 1.1f;
  float frequencyPenalty = 0.0f;
  float presencePenalty = 0.0f;
  std::vector<LlamaToken> repeatPenaltyTokens;
};

class LlamaChat {
 public:
  LlamaChat();
  ~LlamaChat();

  LlamaChat(const LlamaChat&) = delete;
  LlamaChat& operator=(const LlamaChat&) = delete;

  LlamaChat(LlamaChat&&) noexcept = default;
  LlamaChat& operator=(LlamaChat&&) noexcept = default;

  bool InitializeModel(const std::string& modelPath, const ModelParams& params);
  bool InitializeContext(const ContextParams& params);
  void SetSystemPrompt(const std::string& systemPrompt);
  void ResetConversation();

  void Prompt(
      const std::string& userMessage, const SamplingParams& params,
      const std::function<void(const std::string&)>& callback
  );

  [[nodiscard]] std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos = true
  ) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};
