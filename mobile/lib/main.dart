import 'package:flutter/material.dart';

import 'screens/connect_screen.dart';

void main() {
  runApp(const JarvisApp());
}

class JarvisApp extends StatelessWidget {
  const JarvisApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'JARVIS',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        useMaterial3: true,
        // JARVIS vibe: deep navy canvas, cyan/green accents.
        scaffoldBackgroundColor: const Color(0xFF0A0F1E),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00E5C3),
          brightness: Brightness.dark,
        ),
      ),
      home: const ConnectScreen(),
    );
  }
}
