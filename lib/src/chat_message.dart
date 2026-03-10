import 'dart:convert';

/// A message in a chat conversation, following OpenAI's chat message format.
class ChatMessage {
  /// Role: "system", "user", "assistant", or "tool".
  final String role;

  /// Text content (simple string form).
  final String? content;

  /// Multi-modal content parts (alternative to plain string content).
  /// Use for messages with images.
  final List<ContentPart>? contentParts;

  /// Tool calls made by the assistant.
  final List<ToolCall>? toolCalls;

  /// Tool call ID (for "tool" role messages returning results).
  final String? toolCallId;

  const ChatMessage({
    required this.role,
    this.content,
    this.contentParts,
    this.toolCalls,
    this.toolCallId,
  });

  /// Create a system message.
  factory ChatMessage.system(String content) =>
      ChatMessage(role: 'system', content: content);

  /// Create a user text message.
  factory ChatMessage.user(String content) =>
      ChatMessage(role: 'user', content: content);

  /// Create a user multi-modal message with text and images.
  factory ChatMessage.userMultiModal(List<ContentPart> parts) =>
      ChatMessage(role: 'user', contentParts: parts);

  /// Create an assistant message.
  factory ChatMessage.assistant(String content, {List<ToolCall>? toolCalls}) =>
      ChatMessage(role: 'assistant', content: content, toolCalls: toolCalls);

  /// Create a tool result message.
  factory ChatMessage.toolResult(String toolCallId, String content) =>
      ChatMessage(role: 'tool', content: content, toolCallId: toolCallId);

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{'role': role};

    if (contentParts != null && contentParts!.isNotEmpty) {
      map['content'] = contentParts!.map((p) => p.toJson()).toList();
    } else if (content != null) {
      map['content'] = content;
    }

    if (toolCalls != null && toolCalls!.isNotEmpty) {
      map['tool_calls'] = toolCalls!.map((tc) => tc.toJson()).toList();
    }
    if (toolCallId != null) map['tool_call_id'] = toolCallId;

    return map;
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    List<ToolCall>? toolCalls;
    if (json['tool_calls'] is List) {
      toolCalls = (json['tool_calls'] as List)
          .map((tc) => ToolCall.fromJson(tc as Map<String, dynamic>))
          .toList();
    }

    List<ContentPart>? parts;
    String? content;
    if (json['content'] is List) {
      parts = (json['content'] as List)
          .map((p) => ContentPart.fromJson(p as Map<String, dynamic>))
          .toList();
    } else if (json['content'] is String) {
      content = json['content'] as String;
    }

    return ChatMessage(
      role: json['role'] as String? ?? 'user',
      content: content,
      contentParts: parts,
      toolCalls: toolCalls,
      toolCallId: json['tool_call_id'] as String?,
    );
  }
}

/// A content part within a multi-modal message.
class ContentPart {
  final String type; // "text" or "image_url"
  final String? text;
  final ImageUrl? imageUrl;

  const ContentPart({required this.type, this.text, this.imageUrl});

  /// Create a text content part.
  factory ContentPart.text(String text) =>
      ContentPart(type: 'text', text: text);

  /// Create an image content part from a file path.
  factory ContentPart.imageFile(String filePath) => ContentPart(
        type: 'image_url',
        imageUrl: ImageUrl(url: 'file://$filePath'),
      );

  /// Create an image content part from a URL.
  factory ContentPart.imageUrl(String url) =>
      ContentPart(type: 'image_url', imageUrl: ImageUrl(url: url));

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{'type': type};
    if (text != null) map['text'] = text;
    if (imageUrl != null) map['image_url'] = imageUrl!.toJson();
    return map;
  }

  factory ContentPart.fromJson(Map<String, dynamic> json) {
    return ContentPart(
      type: json['type'] as String? ?? 'text',
      text: json['text'] as String?,
      imageUrl: json['image_url'] != null
          ? ImageUrl.fromJson(json['image_url'] as Map<String, dynamic>)
          : null,
    );
  }
}

/// Image URL reference within a content part.
class ImageUrl {
  final String url;
  final String? detail; // "auto", "low", "high"

  const ImageUrl({required this.url, this.detail});

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{'url': url};
    if (detail != null) map['detail'] = detail;
    return map;
  }

  factory ImageUrl.fromJson(Map<String, dynamic> json) {
    return ImageUrl(
      url: json['url'] as String? ?? '',
      detail: json['detail'] as String?,
    );
  }
}

// ---------------------------------------------------------------------------
// Tool calling types
// ---------------------------------------------------------------------------

/// A tool (function) that the model can call.
class Tool {
  final String type; // always "function"
  final FunctionDef function;

  const Tool({this.type = 'function', required this.function});

  Map<String, dynamic> toJson() => {
        'type': type,
        'function': function.toJson(),
      };

  factory Tool.fromJson(Map<String, dynamic> json) {
    return Tool(
      type: json['type'] as String? ?? 'function',
      function:
          FunctionDef.fromJson(json['function'] as Map<String, dynamic>),
    );
  }
}

/// Function definition for a tool.
class FunctionDef {
  final String name;
  final String? description;
  final Map<String, dynamic>? parameters; // JSON Schema

  const FunctionDef({
    required this.name,
    this.description,
    this.parameters,
  });

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{'name': name};
    if (description != null) map['description'] = description;
    if (parameters != null) map['parameters'] = parameters;
    return map;
  }

  factory FunctionDef.fromJson(Map<String, dynamic> json) {
    return FunctionDef(
      name: json['name'] as String? ?? '',
      description: json['description'] as String?,
      parameters: json['parameters'] as Map<String, dynamic>?,
    );
  }
}

/// A tool call made by the assistant in its response.
class ToolCall {
  final String id;
  final String type; // "function"
  final FunctionCall function;

  const ToolCall({required this.id, this.type = 'function', required this.function});

  Map<String, dynamic> toJson() => {
        'id': id,
        'type': type,
        'function': function.toJson(),
      };

  factory ToolCall.fromJson(Map<String, dynamic> json) {
    return ToolCall(
      id: json['id'] as String? ?? '',
      type: json['type'] as String? ?? 'function',
      function:
          FunctionCall.fromJson(json['function'] as Map<String, dynamic>),
    );
  }
}

/// A specific function invocation within a tool call.
class FunctionCall {
  final String name;
  final String arguments; // JSON string

  const FunctionCall({required this.name, required this.arguments});

  /// Parse the arguments JSON string into a map.
  Map<String, dynamic> get parsedArguments =>
      jsonDecode(arguments) as Map<String, dynamic>;

  Map<String, dynamic> toJson() => {
        'name': name,
        'arguments': arguments,
      };

  factory FunctionCall.fromJson(Map<String, dynamic> json) {
    return FunctionCall(
      name: json['name'] as String? ?? '',
      arguments: json['arguments'] as String? ?? '{}',
    );
  }
}
