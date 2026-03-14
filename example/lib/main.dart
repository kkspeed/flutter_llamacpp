import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:flutter_llamacpp/flutter_llamacpp.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const LlamaCppExampleApp());
}

class LlamaCppExampleApp extends StatelessWidget {
  const LlamaCppExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'flutter_llamacpp Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}

// ---------------------------------------------------------------------------
// Model configurations
// ---------------------------------------------------------------------------
class _ModelConfig {
  final String label;
  final String modelUrl;
  final String? mmprojUrl;
  final String modelFilename;
  final String? mmprojFilename;

  const _ModelConfig({
    required this.label,
    required this.modelUrl,
    this.mmprojUrl,
    required this.modelFilename,
    this.mmprojFilename,
  });
}

const _availableModels = [
  _ModelConfig(
    label: 'lft2 · LFM2.5 1.2B Thinking (Q4_K_M)',
    modelUrl:
        'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_K_M.gguf',
    modelFilename: 'LFM2.5-1.2B-Thinking-Q4_K_M.gguf',
  ),
  _ModelConfig(
    label: 'lfm2 · LFM2 350M (Q4_K_M)',
    modelUrl:
        'https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q4_K_M.gguf',
    modelFilename: 'LFM2-350M-Q4_K_M.gguf',
  ),
  _ModelConfig(
    label: 'smollm · SmolLM 135M (Q4_K_M)',
    modelUrl:
        'https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q4_K_M.gguf',
    modelFilename: 'SmolLM-135M.Q4_K_M.gguf',
  ),
  _ModelConfig(
    label: 'Qwen 3.5 0.8B (Q4_K_S)',
    modelUrl:
        'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_S.gguf',
    mmprojUrl:
        'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf',
    modelFilename: 'Qwen3.5-0.8B-Q4_K_S.gguf',
    mmprojFilename: 'mmproj-0.8B-F16.gguf',
  ),
  _ModelConfig(
    label: 'Qwen 3.5 4B (Q4_K_M)',
    modelUrl:
        'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf',
    mmprojUrl:
        'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf',
    modelFilename: 'Qwen3.5-4B-Q4_K_M.gguf',
    mmprojFilename: 'mmproj-4B-F16.gguf',
  ),
];

const _systemPrompt = 'You are a helpful assistant.';
const _contextWindowTokens = 4096;
const _responseReserveTokens = 1024;
const _contextSafetyBufferTokens = 128;
const _imageTokenReserve = 512;

// ---------------------------------------------------------------------------
// Chat Screen
// ---------------------------------------------------------------------------

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _inputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<_ChatBubble> _bubbles = [];

  LlamaModel? _model;
  LlamaEngine? _engine;
  bool _backendInitialized = false;

  String _status = 'Not loaded';
  bool _isLoading = false;
  bool _isGenerating = false;
  double _downloadProgress = 0;
  PerformanceStats? _lastPerf;

  // Model selection
  _ModelConfig _selectedModel = _availableModels[0];

  // Attached images for the next message
  final List<File> _pendingImages = [];

  late String _modelsDirPath;

  @override
  void initState() {
    super.initState();
    _initModelsDir();
  }

  Future<void> _initModelsDir() async {
    final appDir = await getApplicationDocumentsDirectory();
    _modelsDirPath = '${appDir.path}/models';
    await Directory(_modelsDirPath).create(recursive: true);
    _checkExistingModels();
  }

  Future<void> _checkExistingModels() async {
    final modelFile = File('$_modelsDirPath/${_selectedModel.modelFilename}');
    if (await modelFile.exists()) {
      setState(
        () => _status = '${_selectedModel.label} ready. Tap "Load" to start.',
      );
    } else {
      setState(
        () =>
            _status =
                '${_selectedModel.label} not downloaded. Tap "Download" to begin.',
      );
    }
  }

  // -- Download model files --

  Future<void> _downloadFile(String url, String filename) async {
    setState(() {
      _isLoading = true;
      _downloadProgress = 0;
      _status = 'Downloading $filename...';
    });

    try {
      final request = http.Request('GET', Uri.parse(url));
      final response = await http.Client().send(request);
      final contentLength = response.contentLength ?? 0;
      final file = File('$_modelsDirPath/$filename');
      final sink = file.openWrite();

      int received = 0;
      await response.stream
          .listen(
            (chunk) {
              sink.add(chunk);
              received += chunk.length;
              if (contentLength > 0) {
                setState(() => _downloadProgress = received / contentLength);
              }
            },
            onDone: () async {
              await sink.close();
              setState(() {
                _status = '$filename downloaded.';
                _isLoading = false;
              });
            },
            onError: (Object error) async {
              await sink.close();
              setState(() {
                _status = 'Download failed: $error';
                _isLoading = false;
              });
            },
            cancelOnError: true,
          )
          .asFuture();
    } catch (e) {
      setState(() {
        _status = 'Download failed: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _downloadModel() async {
    await _downloadFile(_selectedModel.modelUrl, _selectedModel.modelFilename);
    if (_selectedModel.mmprojUrl != null &&
        _selectedModel.mmprojFilename != null) {
      await _downloadFile(
        _selectedModel.mmprojUrl!,
        _selectedModel.mmprojFilename!,
      );
    }
    setState(
      () => _status = '${_selectedModel.label} ready. Tap "Load" to start.',
    );
  }

  // -- Load model --

  Future<void> _loadModel() async {
    final modelPath = '$_modelsDirPath/${_selectedModel.modelFilename}';
    final mmprojFilename = _selectedModel.mmprojFilename;
    final mmprojPath =
        mmprojFilename == null ? null : '$_modelsDirPath/$mmprojFilename';

    if (!await File(modelPath).exists()) {
      setState(() => _status = 'Model file not found. Please download first.');
      return;
    }

    setState(() {
      _isLoading = true;
      _status = 'Initializing backend...';
    });

    try {
      LlamaModel.initBackend();
      _backendInitialized = true;

      setState(() => _status = 'Loading model...');

      _model = LlamaModel.load(
        modelPath,
        params: const ModelParams(
          nCtx: _contextWindowTokens,
          nBatch: 512,
          nThreads: 4,
          flashAttn: true,
        ),
      );

      if (mmprojPath != null && await File(mmprojPath).exists()) {
        setState(() => _status = 'Loading vision projector...');
        _model!.loadVisionProjector(mmprojPath);
      }

      _engine = LlamaEngine(_model!);

      final info = _model!.info;
      setState(() {
        _status =
            'Loaded: ${info['description'] ?? 'unknown'}\n'
            'Ctx: ${info['n_ctx'] ?? _contextWindowTokens} tokens | '
            'VLM: ${_model!.hasVisionProjector ? 'yes' : 'no'}';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Load failed: $e';
        _isLoading = false;
      });
    }
  }

  // -- Image picker --

  Future<void> _pickImages() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: true,
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() {
        for (final file in result.files) {
          if (file.path != null) {
            _pendingImages.add(File(file.path!));
          }
        }
      });
    }
  }

  void _removePendingImage(int index) {
    setState(() => _pendingImages.removeAt(index));
  }

  /// Resize image to max 1000px on each side, preserving aspect ratio.
  /// Returns the path to the normalized image (or the original if small enough).
  Future<String> _normalizeImage(String inputPath) async {
    const maxDim = 1000;

    final bytes = await File(inputPath).readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;

    final w = image.width;
    final h = image.height;

    // Already within bounds
    if (w <= maxDim && h <= maxDim) {
      image.dispose();
      return inputPath;
    }

    // Compute scaled dimensions
    final scale = (w > h) ? maxDim / w : maxDim / h;
    final newW = (w * scale).round();
    final newH = (h * scale).round();

    // Re-decode at target size
    final resizedCodec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: newW,
      targetHeight: newH,
    );
    final resizedFrame = await resizedCodec.getNextFrame();
    final resizedImage = resizedFrame.image;

    // Encode as PNG
    final byteData = await resizedImage.toByteData(
      format: ui.ImageByteFormat.png,
    );
    resizedImage.dispose();
    image.dispose();

    if (byteData == null) return inputPath;

    // Save to temp file
    final tempDir = await getTemporaryDirectory();
    final name = inputPath.split('/').last.split('.').first;
    final outPath = '${tempDir.path}/norm_${name}_${newW}x$newH.png';
    await File(outPath).writeAsBytes(byteData.buffer.asUint8List());

    return outPath;
  }

  ChatMessage _bubbleToMessage(_ChatBubble bubble) {
    if (bubble.imagePaths.isNotEmpty && _model?.hasVisionProjector == true) {
      return ChatMessage.userMultiModal([
        ContentPart.text(bubble.text),
        ...bubble.imagePaths.map(ContentPart.imageFile),
      ]);
    }
    return ChatMessage(role: bubble.role, content: bubble.text);
  }

  int _extraTokenReserveForMessage(ChatMessage message) {
    final imageCount =
        message.contentParts
            ?.where((part) => part.type == 'image_url')
            .length ??
        0;
    return imageCount * _imageTokenReserve;
  }

  int _extraTokenReserveForMessages(Iterable<ChatMessage> messages) {
    return messages.fold<int>(
      0,
      (total, message) => total + _extraTokenReserveForMessage(message),
    );
  }

  List<ChatMessage> _buildRollingWindowMessages(ChatMessage currentUserMsg) {
    final engine = _engine;
    if (engine == null) {
      return [ChatMessage.system(_systemPrompt), currentUserMsg];
    }

    final history = <ChatMessage>[];
    for (int i = 0; i < _bubbles.length - 2; i++) {
      final bubble = _bubbles[i];
      if (bubble.text.isEmpty) continue;
      history.add(_bubbleToMessage(bubble));
    }

    final systemMessage = ChatMessage.system(_systemPrompt);
    final promptBudget =
        _contextWindowTokens -
        _responseReserveTokens -
        _contextSafetyBufferTokens;
    final selected = <ChatMessage>[currentUserMsg];

    for (int i = history.length - 1; i >= 0; i--) {
      final candidate = <ChatMessage>[systemMessage, history[i], ...selected];
      final request = ChatCompletionRequest(messages: candidate, maxTokens: 1);
      final promptTokens = engine.countPromptTokens(request);
      final totalBudgetCost =
          promptTokens + _extraTokenReserveForMessages(candidate);
      if (totalBudgetCost > promptBudget) {
        continue;
      }
      selected.insert(0, history[i]);
    }

    return [systemMessage, ...selected];
  }

  int _computeMaxResponseTokens(List<ChatMessage> messages) {
    final engine = _engine;
    if (engine == null) return 256;

    final promptTokens = engine.countPromptTokens(
      ChatCompletionRequest(messages: messages, maxTokens: 1),
    );
    final remaining =
        _contextWindowTokens -
        promptTokens -
        _extraTokenReserveForMessages(messages) -
        _contextSafetyBufferTokens;
    return remaining.clamp(64, _responseReserveTokens);
  }

  // -- Send message --

  Future<void> _sendMessage() async {
    final text = _inputController.text.trim();
    if (text.isEmpty || _engine == null || _isGenerating) return;

    // Capture pending images before clearing
    final attachedImages = List<File>.from(_pendingImages);
    final imagePaths = attachedImages.map((f) => f.path).toList();

    _inputController.clear();
    setState(() {
      _pendingImages.clear();
      _bubbles.add(
        _ChatBubble(role: 'user', text: text, imagePaths: imagePaths),
      );
      _bubbles.add(_ChatBubble(role: 'assistant', text: ''));
      _isGenerating = true;
    });
    _scrollToBottom();

    try {
      ChatMessage userMsg;
      if (imagePaths.isNotEmpty && _model?.hasVisionProjector == true) {
        final normalizedPaths = <String>[];
        for (final p in imagePaths) {
          normalizedPaths.add(await _normalizeImage(p));
        }
        final parts = <ContentPart>[
          ContentPart.text(text),
          ...normalizedPaths.map((p) => ContentPart.imageFile(p)),
        ];
        userMsg = ChatMessage.userMultiModal(parts);
      } else {
        userMsg = ChatMessage.user(text);
      }

      final messages = _buildRollingWindowMessages(userMsg);
      final promptTokens = _engine!.countPromptTokens(
        ChatCompletionRequest(messages: messages, maxTokens: 1),
      );
      final droppedMessages = (_bubbles.length - 1) - (messages.length - 1);
      final maxResponseTokens = _computeMaxResponseTokens(messages);

      final request = ChatCompletionRequest(
        messages: messages,
        temperature: 0.7,
        maxTokens: maxResponseTokens,
        enableThinking: _selectedModel.label.startsWith('lft2'),
      );

      final stream = _engine!.chatCompletionStream(request);
      final streamedBuffer = StringBuffer();
      ChatCompletionResponse? response;
      var lastUiFlush = DateTime.now();

      await for (final event in stream) {
        if (event is String) {
          streamedBuffer.write(event);
          final now = DateTime.now();
          if (now.difference(lastUiFlush) >= const Duration(milliseconds: 33) ||
              event.contains('\n')) {
            lastUiFlush = now;
            setState(() {
              _bubbles.last = _ChatBubble(
                role: 'assistant',
                text: streamedBuffer.toString(),
              );
            });
            _scrollToBottom();
          }
        } else if (event is ChatCompletionResponse) {
          response = event;
        }
      }

      final finalText = response?.content ?? streamedBuffer.toString();

      setState(() {
        _bubbles.last = _ChatBubble(
          role: 'assistant',
          text: finalText.isEmpty ? '(no response)' : finalText,
        );
        _isGenerating = false;
        _lastPerf = _engine!.lastPerf;
        _status =
            'Loaded: ${_selectedModel.label}\n'
            'Window: $promptTokens/$_contextWindowTokens prompt tokens, '
            '$maxResponseTokens reserved for reply'
            '${droppedMessages > 0 ? ' | dropped $droppedMessages older messages' : ''}';
      });
      _scrollToBottom();
    } catch (e) {
      setState(() {
        _bubbles.last = _ChatBubble(role: 'assistant', text: 'Error: $e');
        _isGenerating = false;
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _model?.dispose();
    if (_backendInitialized) {
      LlamaModel.shutdownBackend();
    }
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // -- UI --

  @override
  Widget build(BuildContext context) {
    final hasVlm = _model?.hasVisionProjector == true;

    return Scaffold(
      appBar: AppBar(
        title: const Text('flutter_llamacpp'),
        actions: [
          if (_model == null && !_isLoading)
            IconButton(
              icon: const Icon(Icons.download),
              tooltip: 'Download model',
              onPressed: _downloadModel,
            ),
          if (_model == null && !_isLoading)
            IconButton(
              icon: const Icon(Icons.play_arrow),
              tooltip: 'Load model',
              onPressed: _loadModel,
            ),
          if (_isGenerating)
            IconButton(
              icon: const Icon(Icons.stop),
              tooltip: 'Cancel',
              onPressed: () => _engine?.cancel(),
            ),
        ],
      ),
      body: Column(
        children: [
          // Status bar
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Model selector (only when not loaded)
                if (_model == null && !_isLoading)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: DropdownButton<_ModelConfig>(
                      value: _selectedModel,
                      isExpanded: true,
                      isDense: true,
                      dropdownColor:
                          Theme.of(context).colorScheme.surfaceContainerHighest,
                      items:
                          _availableModels.map((m) {
                            return DropdownMenuItem(
                              value: m,
                              child: Text(
                                m.label,
                                style: Theme.of(context).textTheme.bodySmall,
                              ),
                            );
                          }).toList(),
                      onChanged: (value) {
                        if (value != null) {
                          setState(() => _selectedModel = value);
                          _checkExistingModels();
                        }
                      },
                    ),
                  ),
                Text(_status, style: Theme.of(context).textTheme.bodySmall),
                if (_isLoading && _downloadProgress > 0)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: LinearProgressIndicator(value: _downloadProgress),
                  ),
                if (_isLoading && _downloadProgress == 0)
                  const Padding(
                    padding: EdgeInsets.only(top: 4),
                    child: LinearProgressIndicator(),
                  ),
                if (_lastPerf != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      '⚡ ${_lastPerf!}',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.greenAccent,
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Chat messages
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _bubbles.length,
              itemBuilder: (context, index) {
                final bubble = _bubbles[index];
                final isUser = bubble.role == 'user';
                return Align(
                  alignment:
                      isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    constraints: BoxConstraints(
                      maxWidth: MediaQuery.of(context).size.width * 0.8,
                    ),
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color:
                          isUser
                              ? Theme.of(context).colorScheme.primaryContainer
                              : Theme.of(
                                context,
                              ).colorScheme.surfaceContainerHigh,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        // Show attached images
                        if (bubble.imagePaths.isNotEmpty) ...[
                          Wrap(
                            spacing: 6,
                            runSpacing: 6,
                            children:
                                bubble.imagePaths.map((path) {
                                  return ClipRRect(
                                    borderRadius: BorderRadius.circular(8),
                                    child: Image.file(
                                      File(path),
                                      width: 120,
                                      height: 120,
                                      fit: BoxFit.cover,
                                      errorBuilder:
                                          (_, __, ___) => Container(
                                            width: 120,
                                            height: 120,
                                            color: Colors.grey[800],
                                            child: const Icon(
                                              Icons.broken_image,
                                              color: Colors.white54,
                                            ),
                                          ),
                                    ),
                                  );
                                }).toList(),
                          ),
                          const SizedBox(height: 8),
                        ],
                        // Message text or loading indicator
                        if (bubble.text.isEmpty)
                          const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        else
                          SelectableText(
                            bubble.text,
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),

          // Pending image previews
          if (_pendingImages.isNotEmpty)
            Container(
              height: 80,
              padding: const EdgeInsets.symmetric(horizontal: 8),
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _pendingImages.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.all(4),
                    child: Stack(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(
                            _pendingImages[index],
                            width: 68,
                            height: 68,
                            fit: BoxFit.cover,
                          ),
                        ),
                        Positioned(
                          top: -4,
                          right: -4,
                          child: GestureDetector(
                            onTap: () => _removePendingImage(index),
                            child: Container(
                              padding: const EdgeInsets.all(2),
                              decoration: const BoxDecoration(
                                color: Colors.black87,
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(
                                Icons.close,
                                size: 14,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),

          // Input bar
          Container(
            padding: const EdgeInsets.all(8),
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            child: SafeArea(
              child: Row(
                children: [
                  // Image attach button (only when VLM is loaded)
                  if (hasVlm)
                    IconButton(
                      icon: Icon(
                        Icons.image,
                        color:
                            _pendingImages.isNotEmpty
                                ? Theme.of(context).colorScheme.primary
                                : null,
                      ),
                      tooltip: 'Attach image',
                      onPressed:
                          _model != null && !_isGenerating ? _pickImages : null,
                    ),
                  Expanded(
                    child: TextField(
                      controller: _inputController,
                      decoration: InputDecoration(
                        hintText:
                            hasVlm
                                ? 'Message (tap 📷 to add images)...'
                                : 'Type a message...',
                        border: const OutlineInputBorder(),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                      ),
                      enabled: _model != null && !_isGenerating,
                      onSubmitted: (_) => _sendMessage(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    icon: const Icon(Icons.send),
                    onPressed:
                        _model != null && !_isGenerating ? _sendMessage : null,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ChatBubble {
  final String role;
  final String text;
  final List<String> imagePaths;

  const _ChatBubble({
    required this.role,
    required this.text,
    this.imagePaths = const [],
  });
}
