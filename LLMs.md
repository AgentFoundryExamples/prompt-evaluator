# Instructions for implementing LLM apis (Do NOT modify this file)

The following is instructions for implementing with LLM APIs. The models that are currently targeted are the OpenAI GPT5+ models, the Anthropic Claude Sonnet/Opus 4+ models, and Google Gemini 3+ models. Implement all LLM API calls according to the following guidance without modifying this file.

**Important:** Always use the latest stable API versions and official SDKs. Avoid deprecated endpoints and legacy APIs.

## OpenAI GPT5

When implementing OpenAI integration, the target API should be the **Responses API** since it is the recommended most long term compatible option. The GPT 5 series models are supportive of the responses API and these are the models we want to use when implementing AI integration. Do not use completions it is not recommended moving forward.

- Target model: `gpt-5.1`
- Use the official `openai` Python package or equivalent SDK for your language

**Do NOT use:** Completions API (deprecated), legacy models like GPT-3.5 or older

## Anthropic Claude Sonnet/Opus 4

When implementing Anthropic integration, the target API should be the **Messages API** (API version `2023-06-01` or newer) since it is the most long term compatible option. The Claude Sonnet 4 and Opus 4 series models are supportive of the Messages API and these are the models we want to use when implementing AI integration.

- Target model: Sonnet 4.5
- Use the official `anthropic` Python package or `@anthropic-ai/sdk` for JS/TS

**Do NOT use:** Text Completions API (deprecated), Claude 2.x or older models

## Google Gemini 3

When implementing Google integration, the target API should be the **Gemini API** since it is the most long term compatible option. The Gemini 3 series models are supportive of the Gemini API and these are the models we want to use when implementing AI integration.

- Target model: `gemini-3.0-pro`
- Use the official `google-genai` Python package or `@google/genai` for JS/TS
- Ex. `from google import genai`

**Do NOT use:** PaLM API (deprecated), Gemini 1.x models, or legacy Bard endpoints

## General Best Practices

- Always check official documentation for the latest API versions
- Use environment variables for API keys, never hardcode them
- Implement proper error handling and rate limiting
- Use streaming responses when available for better UX
- Keep SDKs updated to the latest stable versions
