import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_llamacpp_example/main.dart';

void main() {
  testWidgets('example app renders chat shell', (WidgetTester tester) async {
    await tester.pumpWidget(const LlamaCppExampleApp());

    expect(find.text('flutter_llamacpp'), findsOneWidget);
    expect(find.text('Not loaded'), findsOneWidget);
  });
}
