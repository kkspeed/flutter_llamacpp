import 'dart:async';

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
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
// Model download URLs
// ---------------------------------------------------------------------------
const _modelUrl =
    'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_S.gguf';
const _mmprojUrl =
    'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf';
const _modelFilename = 'Qwen3.5-0.8B-Q4_K_S.gguf';
const _mmprojFilename = 'mmproj-F16.gguf';

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

  String _status = 'Not loaded';
  bool _isLoading = false;
  bool _isGenerating = false;
  double _downloadProgress = 0;
  PerformanceStats? _lastPerf;


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
    final modelFile = File('$_modelsDirPath/$_modelFilename');
    if (await modelFile.exists()) {
      setState(() => _status = 'Model ready. Tap "Load" to start.');
    } else {
      setState(() => _status = 'Model not downloaded. Tap "Download" to begin.');
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
      await response.stream.listen(
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
      ).asFuture();
    } catch (e) {
      setState(() {
        _status = 'Download failed: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _downloadModel() async {
    await _downloadFile(_modelUrl, _modelFilename);
    await _downloadFile(_mmprojUrl, _mmprojFilename);
    setState(() => _status = 'Models ready. Tap "Load" to start.');
  }

  // -- Load model --

  Future<void> _loadModel() async {
    final modelPath = '$_modelsDirPath/$_modelFilename';
    final mmprojPath = '$_modelsDirPath/$_mmprojFilename';

    if (!await File(modelPath).exists()) {
      setState(() => _status = 'Model file not found. Please download first.');
      return;
    }

    setState(() {
      _isLoading = true;
      _status = 'Initializing backend...';
    });

    try {
      // Init backend
      LlamaModel.initBackend();

      setState(() => _status = 'Loading model...');

      // Load model (heavy operation)
      _model = LlamaModel.load(
        modelPath,
        params: const ModelParams(
          nCtx: 4096,
          nBatch: 512,
          nThreads: 4,
          flashAttn: true,
        ),
      );

      // Load mmproj if available
      if (await File(mmprojPath).exists()) {
        setState(() => _status = 'Loading vision projector...');
        _model!.loadVisionProjector(mmprojPath);
      }

      _engine = LlamaEngine(_model!);

      final info = _model!.info;
      setState(() {
        _status = 'Loaded: ${info['description'] ?? 'unknown'}\n'
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

  // -- Send message --

  Future<void> _sendMessage() async {
    final text = _inputController.text.trim();
    if (text.isEmpty || _engine == null || _isGenerating) return;

    _inputController.clear();
    setState(() {
      _bubbles.add(_ChatBubble(role: 'user', text: text));
      _bubbles.add(_ChatBubble(role: 'assistant', text: ''));
      _isGenerating = true;
    });
    _scrollToBottom();

    try {
      // Build messages list
      final messages = <ChatMessage>[
        ChatMessage.system('You are a helpful assistant.'),
        ...(_bubbles
            .where((b) => b.text.isNotEmpty)
            .map((b) => ChatMessage(role: b.role, content: b.text))),
      ];

      final request = ChatCompletionRequest(
        messages: messages,
        temperature: 0.7,
        maxTokens: 65536,
      );

      // Use non-streaming for now
      final response = await _engine!.chatCompletion(request);

      setState(() {
        _bubbles.last = _ChatBubble(
          role: 'assistant',
          text: response.content ?? '(no response)',
        );
        _isGenerating = false;
        _lastPerf = _engine!.lastPerf;
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
    LlamaModel.shutdownBackend();
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // -- UI --

  @override
  Widget build(BuildContext context) {
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
                      color: isUser
                          ? Theme.of(context).colorScheme.primaryContainer
                          : Theme.of(context).colorScheme.surfaceContainerHigh,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: bubble.text.isEmpty
                        ? const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : SelectableText(
                            bubble.text,
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
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
                  Expanded(
                    child: TextField(
                      controller: _inputController,
                      decoration: const InputDecoration(
                        hintText: 'Type a message...',
                        border: OutlineInputBorder(),
                        contentPadding: EdgeInsets.symmetric(
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
                    onPressed: _model != null && !_isGenerating
                        ? _sendMessage
                        : null,
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

  const _ChatBubble({required this.role, required this.text});
}
