import 'chat_message.dart';

/// Chat completion request, following OpenAI's API format.
class ChatCompletionRequest {
  final List<ChatMessage> messages;
  final double temperature;
  final double topP;
  final int topK;
  final double minP;
  final int maxTokens;
  final List<Tool>? tools;
  final String? toolChoice; // "auto", "none", "required"
  final bool stream;

  const ChatCompletionRequest({
    required this.messages,
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    this.minP = 0.0,
    this.maxTokens = 512,
    this.tools,
    this.toolChoice,
    this.stream = true,
  });

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{
      'messages': messages.map((m) => m.toJson()).toList(),
      'temperature': temperature,
      'top_p': topP,
      'top_k': topK,
      'min_p': minP,
      'max_tokens': maxTokens,
      'stream': stream,
    };
    if (tools != null && tools!.isNotEmpty) {
      map['tools'] = tools!.map((t) => t.toJson()).toList();
    }
    if (toolChoice != null) map['tool_choice'] = toolChoice;
    return map;
  }
}

/// Chat completion response, following OpenAI's API format.
class ChatCompletionResponse {
  final String id;
  final String object;
  final List<Choice> choices;
  final Usage usage;

  const ChatCompletionResponse({
    required this.id,
    required this.object,
    required this.choices,
    required this.usage,
  });

  factory ChatCompletionResponse.fromJson(Map<String, dynamic> json) {
    return ChatCompletionResponse(
      id: json['id'] as String? ?? '',
      object: json['object'] as String? ?? 'chat.completion',
      choices: (json['choices'] as List? ?? [])
          .map((c) => Choice.fromJson(c as Map<String, dynamic>))
          .toList(),
      usage: Usage.fromJson(json['usage'] as Map<String, dynamic>? ?? {}),
    );
  }

  /// Convenience: get the first choice's message content.
  String? get content => choices.isNotEmpty ? choices.first.message.content : null;

  /// Convenience: get tool calls from the first choice.
  List<ToolCall>? get toolCalls => choices.isNotEmpty ? choices.first.message.toolCalls : null;
}

/// A choice in the completion response.
class Choice {
  final int index;
  final ChatMessage message;
  final String finishReason;

  const Choice({
    required this.index,
    required this.message,
    required this.finishReason,
  });

  factory Choice.fromJson(Map<String, dynamic> json) {
    return Choice(
      index: json['index'] as int? ?? 0,
      message: ChatMessage.fromJson(json['message'] as Map<String, dynamic>? ?? {}),
      finishReason: json['finish_reason'] as String? ?? 'stop',
    );
  }
}

/// Token usage statistics.
class Usage {
  final int promptTokens;
  final int completionTokens;
  final int totalTokens;

  const Usage({
    required this.promptTokens,
    required this.completionTokens,
    required this.totalTokens,
  });

  factory Usage.fromJson(Map<String, dynamic> json) {
    return Usage(
      promptTokens: json['prompt_tokens'] as int? ?? 0,
      completionTokens: json['completion_tokens'] as int? ?? 0,
      totalTokens: json['total_tokens'] as int? ?? 0,
    );
  }
}

/// Performance statistics from a generation.
class PerformanceStats {
  final double promptTokensPerSec;
  final double genTokensPerSec;
  final int promptTokens;
  final int genTokens;
  final double promptMs;
  final double genMs;

  const PerformanceStats({
    required this.promptTokensPerSec,
    required this.genTokensPerSec,
    required this.promptTokens,
    required this.genTokens,
    required this.promptMs,
    required this.genMs,
  });

  factory PerformanceStats.fromJson(Map<String, dynamic> json) {
    return PerformanceStats(
      promptTokensPerSec: (json['prompt_tokens_per_sec'] as num?)?.toDouble() ?? 0.0,
      genTokensPerSec: (json['gen_tokens_per_sec'] as num?)?.toDouble() ?? 0.0,
      promptTokens: json['prompt_tokens'] as int? ?? 0,
      genTokens: json['gen_tokens'] as int? ?? 0,
      promptMs: (json['prompt_ms'] as num?)?.toDouble() ?? 0.0,
      genMs: (json['gen_ms'] as num?)?.toDouble() ?? 0.0,
    );
  }

  @override
  String toString() =>
      'Prompt: ${promptTokensPerSec.toStringAsFixed(1)} t/s ($promptTokens tokens, ${promptMs.toStringAsFixed(0)}ms) | '
      'Gen: ${genTokensPerSec.toStringAsFixed(1)} t/s ($genTokens tokens, ${genMs.toStringAsFixed(0)}ms)';
}
