#pragma once

#include <string>
#include <memory>
#include <functional>

class LlamaWrapper {
 public:
  LlamaWrapper();
  ~LlamaWrapper();

  LlamaWrapper(const LlamaWrapper&) = delete;
  LlamaWrapper& operator=(const LlamaWrapper&) = delete;

  LlamaWrapper(LlamaWrapper&&) noexcept;
  LlamaWrapper& operator=(LlamaWrapper&&) noexcept;

  bool Initialize(const std::string& model_path, size_t context_size = 80000);

  std::string RunQuery(const std::string& prompt, size_t max_tokens = 1000);

  void RunQueryStream(const std::string& prompt, size_t max_tokens, const std::function<void(const std::string&)>& callback);

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};
