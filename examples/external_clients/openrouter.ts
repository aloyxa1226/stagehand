/**
 * Welcome to the Stagehand OpenRouter client!
 *
 * This is a client for the OpenRouter API. It is a wrapper around the OpenAI API
 * that allows you to create chat completions with OpenRouter's models.
 *
 * To use this client, you need to have an OpenRouter API key. You can get one at:
 * https://openrouter.ai/keys
 */

import { CreateChatCompletionOptions, LLMClient } from "@/dist";
import { LogLine } from "../../types/log";
import { LLMCache } from "../../lib/cache/LLMCache";
import OpenAI, { type ClientOptions } from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartText,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
} from "openai/resources/chat/completions";
import { z } from "zod";

function validateZodSchema(schema: z.ZodTypeAny, data: unknown) {
  try {
    schema.parse(data);
    return true;
  } catch {
    return false;
  }
}

export class OpenRouterClient extends LLMClient {
  public type = "openrouter" as const;
  private client: OpenAI;
  private cache: LLMCache | undefined;
  private enableCaching: boolean;
  public clientOptions: ClientOptions;
  public modelName: "google/gemini-2.0-pro-exp-02-05:free";
  public hasVision = false;

  constructor({
    enableCaching = false,
    cache,
    modelName,
    clientOptions,
  }: {
    logger: (message: LogLine) => void;
    enableCaching?: boolean;
    cache?: LLMCache;
    modelName: "google/gemini-2.0-pro-exp-02-05:free";
    clientOptions?: ClientOptions;
  }) {
    super(modelName);
    this.clientOptions = clientOptions;
    this.client = new OpenAI({
      ...clientOptions,
      baseURL: "https://openrouter.ai/api/v1",
    });
    this.cache = cache;
    this.enableCaching = enableCaching;
    this.modelName = modelName;
  }

  async createChatCompletion<T = ChatCompletion>({
    options,
    retries = 3,
    logger,
  }: CreateChatCompletionOptions): Promise<T> {
    const { image, requestId, ...optionsWithoutImageAndRequestId } = options;

    logger({
      category: "openrouter",
      message: "creating chat completion",
      level: 1,
      auxiliary: {
        options: {
          value: JSON.stringify({
            ...optionsWithoutImageAndRequestId,
            requestId,
          }),
          type: "object",
        },
        modelName: {
          value: this.modelName,
          type: "string",
        },
      },
    });

    const cacheOptions = {
      model: this.modelName,
      messages: options.messages,
      temperature: options.temperature,
      top_p: options.top_p,
      frequency_penalty: options.frequency_penalty,
      presence_penalty: options.presence_penalty,
      image: image,
      response_model: options.response_model,
    };

    if (this.enableCaching) {
      const cachedResponse = await this.cache.get<T>(
        cacheOptions,
        options.requestId,
      );
      if (cachedResponse) {
        logger({
          category: "llm_cache",
          message: "LLM cache hit - returning cached response",
          level: 1,
          auxiliary: {
            requestId: {
              value: options.requestId,
              type: "string",
            },
            cachedResponse: {
              value: JSON.stringify(cachedResponse),
              type: "object",
            },
          },
        });
        return cachedResponse;
      }
    }

    let responseFormat = undefined;
    if (options.response_model) {
      // For Gemini models, we need to handle the schema differently
      if (this.modelName.startsWith("google/gemini")) {
        // Add the schema as a system message instead of using responseFormat
        options.messages.push({
          role: "system",
          content: `Please format your response according to this schema:\n${JSON.stringify(options.response_model.schema)}\n\nRespond with only the JSON object, no additional text or formatting.`,
        });
      } else {
        responseFormat = zodResponseFormat(
          options.response_model.schema,
          options.response_model.name,
        );
      }
    }

    const openRouterOptions = {
      ...optionsWithoutImageAndRequestId,
      model: this.modelName as string,
    };

    logger({
      category: "openrouter",
      message: "creating chat completion",
      level: 1,
      auxiliary: {
        openRouterOptions: {
          value: JSON.stringify(openRouterOptions),
          type: "object",
        },
      },
    });

    const formattedMessages: ChatCompletionMessageParam[] =
      options.messages.map((message) => {
        if (Array.isArray(message.content)) {
          const contentParts = message.content.map((content) => {
            if ("image_url" in content) {
              const imageContent: ChatCompletionContentPartImage = {
                image_url: {
                  url: content.image_url.url,
                },
                type: "image_url",
              };
              return imageContent;
            } else {
              const textContent: ChatCompletionContentPartText = {
                text: content.text,
                type: "text",
              };
              return textContent;
            }
          });

          if (message.role === "system") {
            const formattedMessage: ChatCompletionSystemMessageParam = {
              ...message,
              role: "system",
              content: contentParts.filter(
                (content): content is ChatCompletionContentPartText =>
                  content.type === "text",
              ),
            };
            return formattedMessage;
          } else if (message.role === "user") {
            const formattedMessage: ChatCompletionUserMessageParam = {
              ...message,
              role: "user",
              content: contentParts,
            };
            return formattedMessage;
          } else {
            const formattedMessage: ChatCompletionAssistantMessageParam = {
              ...message,
              role: "assistant",
              content: contentParts.filter(
                (content): content is ChatCompletionContentPartText =>
                  content.type === "text",
              ),
            };
            return formattedMessage;
          }
        }

        const formattedMessage: ChatCompletionUserMessageParam = {
          role: "user",
          content: message.content,
        };

        return formattedMessage;
      });

    const body: ChatCompletionCreateParamsNonStreaming = {
      ...openRouterOptions,
      model: this.modelName,
      messages: formattedMessages,
      response_format: responseFormat,
      stream: false,
      tools: options.tools?.map((tool) => ({
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
        type: "function",
      })),
    };

    const response = await this.client.chat.completions.create(body);

    logger({
      category: "openrouter",
      message: "response",
      level: 1,
      auxiliary: {
        response: {
          value: JSON.stringify(response),
          type: "object",
        },
        requestId: {
          value: requestId,
          type: "string",
        },
      },
    });

    if (!response?.choices?.[0]?.message) {
      if (retries > 0) {
        return this.createChatCompletion({
          options,
          logger,
          retries: retries - 1,
        });
      }
      throw new Error("Invalid or empty response from OpenRouter API");
    }

    if (options.response_model) {
      const extractedData = response.choices[0].message.content;
      if (!extractedData) {
        if (retries > 0) {
          return this.createChatCompletion({
            options,
            logger,
            retries: retries - 1,
          });
        }
        throw new Error("No content in response");
      }

      try {
        // Strip markdown code block formatting if present
        const cleanedContent = extractedData
          .replace(/^\s*```(?:json)?\n?/g, "") // Remove opening code block
          .replace(/\n?```\s*$/g, "") // Remove closing code block
          .trim();

        const parsedData = JSON.parse(cleanedContent);
        if (!validateZodSchema(options.response_model.schema, parsedData)) {
          if (retries > 0) {
            return this.createChatCompletion({
              options,
              logger,
              retries: retries - 1,
            });
          }
          throw new Error("Response does not match the required schema");
        }

        if (this.enableCaching) {
          this.cache.set(cacheOptions, parsedData, options.requestId);
        }

        return parsedData;
      } catch (error) {
        logger({
          category: "openrouter",
          message: "Failed to parse response",
          level: 0,
          auxiliary: {
            error: {
              value: error.message,
              type: "string",
            },
            content: {
              value: extractedData,
              type: "string",
            },
          },
        });

        if (retries > 0) {
          return this.createChatCompletion({
            options,
            logger,
            retries: retries - 1,
          });
        }
        throw new Error(`Failed to parse response: ${error.message}`);
      }
    }

    if (this.enableCaching) {
      this.cache.set(cacheOptions, response, options.requestId);
    }

    return response as T;
  }
}
