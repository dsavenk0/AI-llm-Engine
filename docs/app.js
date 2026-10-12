/**
 * AI-llm-Engine & LLMBar — Interactive Client Script
 * Apple Silicon + Cyberpunk HUD Theme & Localization Controller
 */

(function () {
  'use strict';

  // ==========================================
  // 1. Localization Dictionary (EN & RU)
  // ==========================================
  const translations = {
    en: {
      nav_overview: "Overview",
      nav_llmbar: "LLMBar HUD",
      nav_gallery: "Screenshots",
      nav_architecture: "Architecture",
      nav_benchmarks: "Benchmarks",
      nav_models: "Models",
      nav_install: "Install",
      nav_download: "Download",

      theme_dark: "Dark HUD",
      theme_light: "Light Glass",

      hero_eyebrow: "Apple Silicon M1–M5 Native · Hadamard Ternary · HUD Telemetry",
      hero_title_1: "Extreme intelligence.",
      hero_title_2: "Minus the overhead.",
      hero_sub: "A dependency-light MLX inference engine with Hadamard ternary quantization, prompt-lookup speculative decoding, and a native menu bar companion. Run 27B parameters on a fanless 16 GB MacBook Air at up to 35 tokens per second.",
      hero_download_dmg: "Download LLMBar (.dmg)",
      hero_cli: "Terminal Quickstart",
      copied: "Copied",

      stat_spec_title: "Speculative Decoding",
      stat_spec_desc: "2.5× faster code edits and tool calls via exact prompt-lookup verification.",
      stat_cold_title: "Instant Cold Start",
      stat_cold_desc: "From zero to first token in 4 seconds. 4.7× faster than standard vMLX.",
      stat_ram_title: "27B in Unified Memory",
      stat_ram_desc: "Hadamard ternary weights stay pinned. Zero disk swapping.",
      stat_proc_title: "Featherweight Runtime",
      stat_proc_desc: "48% less process overhead than competing engines.",

      showcase_kicker: "// NATIVE HUD COMPANION",
      showcase_headline: "A precision instrument in your menu bar.",
      showcase_sub: "Built entirely in Swift with zero third-party dependencies. Telemetry, RAM protection, model switching, and local OpenAI API orchestration — right under your notch.",
      engine_active: "ACTIVE",

      tab_telemetry: "Telemetry",
      tab_speculation: "Speculation (35 tok/s)",
      tab_activity: "Activity & Memory",
      tab_terminal: "Terminal Chat",
      tab_integrations: "Integrations",

      hud_throughput: "DECODE THROUGHPUT",
      hud_spec_active: "PROMPT-LOOKUP DRAFTING (×2.5)",
      hud_prefill: "Prefill Speed",
      hud_cold_start: "Cold to First Token",
      hud_prefix_cache: "SSD Prefix Reused",
      hud_memory_guard: "RAM GUARD & WIRED MEMORY",
      hud_unified_status: "Unified Memory Pressure: <strong>Nominal</strong>",
      hud_weights: "Weights (7.6G)",
      hud_ssm: "SSM (0.4G)",
      hud_proc: "Process (0.6G)",
      hud_headroom: "Headroom (3.5G)",
      hud_model_name: "Active Model",
      hud_quant: "Quantization",
      hud_guard_ceiling: "Memory Ceiling",
      hud_sleep_status: "Auto-Sleep",

      spec_title: "Prompt-Lookup Speculative Decoding",
      spec_sub: "When code edits repeat context, the engine drafts 15-gram continuations and verifies them in a single weight pass. Identical greedy output, 2.5× the speed.",
      spec_run: "Run Speculation Test",

      act_thermal: "THERMAL STATE",
      act_swap: "DISK SWAPPING",
      act_ssd: "SSD KV SNAPSHOTS",
      act_context: "COMPUTED CONTEXT",

      integ_head: "Seamless OpenAI API Compatibility",
      integ_sub_desc: "AI-llm-Engine serves on http://127.0.0.1:8000/v1. Drop-in ready for Continue, Kilo Code, OpenCode, Claude Code, and Cursor with automatic max_tokens clamping.",
      copy_config: "Copy Configuration",
      copy_url: "Copy Base URL",

      gallery_kicker: "// NATIVE APP SCREENSHOTS",
      gallery_headline: "Built for macOS. Inside and out.",
      gallery_sub: "High-resolution captures of LLMBar running on macOS Sonoma & Sequoia. Cyberpunk HUD styling, real-time memory telemetry, and instant model orchestration.",
      shot_1_badge: "ENGINE TAB · 35 TOK/S",
      shot_1_title: "Core Telemetry & Throughput",
      shot_1_desc: "Live decode speed (34.8 tok/s), prefill rate (119.5 tok/s), unified memory guard, and single-pass Hadamard verification.",
      shot_2_badge: "MONITOR TAB · RAM GUARD",
      shot_2_title: "Hardware & Unified Memory Pressure",
      shot_2_desc: "Detailed Activity Monitor breakdown (Weights, SSM, Process, Headroom), 0-byte swap guarantee, and nominal thermal tracking.",
      shot_3_badge: "MODELS TAB · HUGGING FACE",
      shot_3_title: "Hugging Face Model Manager",
      shot_3_desc: "Search, pull, and switch models with 1 click. Immediate RAM suitability check and auto-engine restart.",
      shot_4_badge: "LLM EDITOR · CHAT",
      shot_4_title: "An Agent That Asks Before It Edits",
      shot_4_desc: "Chat-first layout with Manual, Edit, Plan and Auto modes: the diff of every change is shown and nothing is written until you approve.",
      shot_5_badge: "LLM EDITOR · EDITOR",
      shot_5_title: "A Native Editor Next to the Assistant",
      shot_5_desc: "File tree, syntax colours, problems, terminal and git with per-file discard, in a small Swift app that runs only while you use it.",

      arch_kicker: "// SILICON ARCHITECTURE",
      arch_headline: "Why it’s so fast on Apple Silicon.",
      arch_sub: "Built from the ground up to take advantage of Apple Unified Memory Architecture and Metal compilation.",
      feat1_title: "One Hadamard Transform per Shared Input",
      feat1_desc: "The bundle stores one sign vector per input width, verified on load. Projections q|k|v, gate|up and qkv|z execute as a single fused quantized matmul after one transform. Competing engines perform a redundant transform for every projection.",
      feat2_title: "Prompt-Lookup Speculation",
      feat2_desc: "Drafts continuation n-grams during code edits and verifies up to 15 drafted tokens in one weight pass. Verifying 15 tokens costs ~5 decode steps. Shuts off automatically when prose doesn’t match.",
      feat3_title: "Pinned Wired Weights",
      feat3_desc: "The wired memory limit keeps model weights resident in physical RAM. macOS is strictly prevented from compressing or swapping them to disk, eliminating system stutter.",
      feat4_title: "Hybrid SSD Prefix Cache",
      feat4_desc: "GatedDeltaNet recurrence states and KV caches are snapshotted to NVMe SSD at prompt completion. Subsequent agent requests restore instantly without recomputing system prompts.",
      feat5_title: "Adaptive RAM Guard",
      feat5_desc: "Computes safe context length dynamically. Caps prefill chunk sizes (1024 → 256 → 128) and enforces an 88% system memory ceiling so other apps keep running smoothly.",
      feat6_title: "15-Minute Zero-Footprint Sleep",
      feat6_desc: "After 15 minutes of idle time, the engine unloads model weights and returns ~7.6 GB of unified memory back to macOS. Wakes up in approximately 2 seconds on the next incoming request.",
      feat7_title: "1.75-bit Ternary on One Metal Kernel",
      feat7_desc: "PTQ1_0, TQ1_G128 and JANG-packed ternary bundles stay packed in memory and run on a lookup-table Metal kernel. A 27B model fits in ~6 GB; everything else MLX goes through the optional mlx-lm backend.",
      feat8_title: "A Native Code Editor, On Demand",
      feat8_desc: "A separate Swift app — never loaded until you open it: chat-first and editor layouts, an agent with Manual / Edit / Plan / Auto modes and a reasoning-effort control, start and stop the model from the toolbar, git with per-file discard, GitHub, terminal, problems, a preview pane, dictation and a context ring that compacts the chat. Dark and light themes.",
      feat9_title: "Optional Modules, Off by Default",
      feat9_desc: "ModelScope as a second model source, pictures drawn right in the chat through mflux, web search, Open VSX themes and snippets, activation edits, another OpenAI-compatible server as the assistant's model, proxy and CA settings. None of it runs until you ask.",
      feat10_title: "GGUF on the Engine's Own Decoder",
      feat10_desc: "Llama, Qwen2 and Qwen3 GGUF files are repacked into MLX quantization on load and run by the engine itself — the same server, prompt cache and speculation as everything else, with logits that match llama.cpp's. Anything else falls back to llama.cpp, downloaded only when needed.",

      perf_kicker: "// VERIFIED BENCHMARKS",
      perf_headline: "Real numbers. No cherry-picking.",
      perf_sub: "Benchmarked on a fanless MacBook Air M5 with 16 GB unified RAM running dealignai/Bonsai-2-27B-CRACK-Ternary-JANG (27B, ~7.6 GB). Alternating runs with identical prompts.",
      perf_metric: "Benchmark Metric",
      perf_diff: "Performance Delta",
      p_decode: "Decode Speed (256 tokens)",
      p_decode_sub: "Pure autoregressive decoding",
      p_spec: "Code Edits (Speculative Decoding)",
      p_spec_sub: "Prompt-lookup n-gram matches",
      p_prefill: "Prefill Speed (2.3K prompt)",
      p_prefill_sub: "Prompt ingestion rate",
      p_cold: "Cold Start to First Request",
      p_cold_sub: "Load, compile & generate",
      p_overhead: "Process Memory Overhead",
      p_overhead_sub: "RAM used above the weights",
      p_size: "Install Footprint",
      p_size_sub: "Application / environment size",
      p_tokens: "Large max_tokens (e.g. 32K)",
      p_tokens_sub: "Agent requests from Kilo / Continue",

      mod_kicker: "// HARDWARE SELECTOR",
      mod_headline: "Choose your model. Match your Mac.",
      mod_sub: "Pick your Mac’s unified memory configuration to see recommended models and expected throughput.",
      ram_label: "Unified Memory:",
      mod_best_16: "Recommended for 16GB",
      mod_pro_tier: "Requires 24GB+",
      mod_entry: "Entry Tier",
      mod_bonsai_desc: "27B parameter hybrid dense/SSM model. Quantized with Hadamard rotation. Runs comfortably inside 7.6 GB RAM.",
      mod_35b_desc: "MoE architecture with Hadamard ternary rotation. Highest reasoning capability for large-context programming agents.",
      mod_4b_desc: "Ultra-lightweight model ideal for quick chats, summary tasks, and 8 GB Macs. Loads in under one second.",

      inst_kicker: "// DEPLOYMENT",
      inst_headline: "Ready in minutes.",
      inst_sub: "Download the native DMG installer or setup with the standalone CLI.",
      inst_rec: "RECOMMENDED",
      inst_dmg_title: "macOS Installer DMG",
      inst_dmg_desc: "Drag-and-drop installer containing LLMBar.app, bundled CLI binaries, and menu bar daemon.",
      step1: "Download LLMBar-1.0.dmg from GitHub Releases.",
      step2: "Drag LLMBar.app to your Applications folder.",
      step3: "Launch from Applications or Menu Bar. Done!",
      btn_download_now: "Download from GitHub Releases",
      inst_cli_tag: "DEVELOPER CLI",
      inst_cli_title: "Terminal & Python CLI",
      inst_cli_desc: "For developers who prefer direct terminal access, script automation, and isolated venvs.",
      copy_cli: "Copy Terminal Setup Command",

      faq_kicker: "// INTEL DATABASE",
      faq_headline: "Frequently Asked Questions",
      faq_q1: "What makes AI-llm-Engine faster than Ollama or vMLX?",
      faq_a1: "AI-llm-Engine applies Hadamard sign vectors once per shared input dimension rather than per projection, fusing them into Metal matmuls. In addition, prompt-lookup speculative decoding drafts n-grams to achieve 27–35 tok/s on code repetitions, and cold-start compile caches drop startup time from 19 seconds down to 4 seconds.",
      faq_q2: "Can a 16 GB MacBook Air really run a 27-billion parameter model?",
      faq_a2: "Yes. With Hadamard ternary quantization (JANG / Prism), the 27B weights require only ~7.6 GB of unified memory. Process overhead is capped at 0.66 GB, and the RAM Guard ceiling keeps total system memory under 88%, preventing macOS from swapping to disk or stuttering.",
      faq_q3: "How do I connect this to Continue.dev or Cursor?",
      faq_a3: "The engine runs a standard OpenAI-compatible HTTP server at http://127.0.0.1:8000/v1. In Continue.dev, configure provider as 'openai' with model 'default' and apiBase pointing to this address. Any OpenAI client works out of the box.",
      faq_q4: "What are the macOS system requirements?",
      faq_a4: "Any Mac with Apple Silicon (M1, M2, M3, M4, or M5) running macOS 14.0 (Sonoma) or macOS 15.0+ (Sequoia). Works with as little as 8 GB RAM (for 4B models) up to 16 GB, 24 GB, 36 GB, 64 GB, 128 GB+.",

      cta_title: "Run intelligence at the speed of silicon.",
      cta_sub: "Download LLMBar.app today. 100% on-device, private, and open source under Apache 2.0.",
      cta_download: "Download on GitHub Releases",
      cta_github: "GitHub Repository"
    },

    ru: {
      nav_overview: "Обзор",
      nav_llmbar: "LLMBar HUD",
      nav_gallery: "Скриншоты",
      nav_architecture: "Архитектура",
      nav_benchmarks: "Бенчмарки",
      nav_models: "Модели",
      nav_install: "Установка",
      nav_download: "Скачать",

      theme_dark: "Темная HUD",
      theme_light: "Светлое стекло",

      hero_eyebrow: "Apple Silicon M1–M5 Native · Квантование Адамара · HUD Телеметрия",
      hero_title_1: "Колоссальная мощь.",
      hero_title_2: "Ноль компромиссов.",
      hero_sub: "Легковесный MLX-движок с тернарным квантованием Адамара, спекулятивным декодированием и нативным приложением для меню macOS. Запускайте 27B модели на MacBook Air 16 GB на скорости до 35 токенов в секунду.",
      hero_download_dmg: "Скачать LLMBar (.dmg)",
      hero_cli: "Установка через терминал",
      copied: "Скопировано",

      stat_spec_title: "Спекулятивное декодирование",
      stat_spec_desc: "В 2.5 раза быстрее для правок кода благодаря проверке n-грамм контекста.",
      stat_cold_title: "Холодный старт за 4 сек",
      stat_cold_desc: "От запуска до первого токена за 4 секунды. В 4.7 раза быстрее vMLX.",
      stat_ram_title: "27B в объединенной памяти",
      stat_ram_desc: "Тернарные веса закреплены в памяти. Полное отсутствие свопа macOS.",
      stat_proc_title: "Минимальный оверхед",
      stat_proc_desc: "На 48% меньше накладных расходов памяти, чем у других движков.",

      showcase_kicker: "// НАЦИОНАЛЬНЫЙ HUD КОМПАНЬОН",
      showcase_headline: "Прецизионный инструмент в строке меню.",
      showcase_sub: "Написан на чистом Swift без сторонних зависимостей. Телеметрия в реальном времени, защита RAM, переключение моделей и локальный сервер OpenAI прямо под вырезом экрана.",
      engine_active: "АКТИВЕН",

      tab_telemetry: "Телеметрия",
      tab_speculation: "Спекуляция (35 tok/s)",
      tab_activity: "Монитор и Память",
      tab_terminal: "Чат в Терминале",
      tab_integrations: "Интеграции",

      hud_throughput: "ПРОПУСКНАЯ СПОСОБНОСТЬ",
      hud_spec_active: "ПРОМПТ-СПЕКУЛЯЦИЯ (×2.5)",
      hud_prefill: "Скорость префилла",
      hud_cold_start: "Холодный старт",
      hud_prefix_cache: "Повторно из SSD",
      hud_memory_guard: "СТОРОЖ RAM И ПАМЯТЬ",
      hud_unified_status: "Давление объединенной памяти: <strong>В норме</strong>",
      hud_weights: "Веса (7.6G)",
      hud_ssm: "SSM (0.4G)",
      hud_proc: "Процесс (0.6G)",
      hud_headroom: "Запас (3.5G)",
      hud_model_name: "Текущая модель",
      hud_quant: "Квантование",
      hud_guard_ceiling: "Лимит памяти",
      hud_sleep_status: "Авто-сон",

      spec_title: "Спекулятивное декодирование по контексту",
      spec_sub: "Когда генерация повторяет контекст (редактирование кода, цитаты), движок берет n-граммы и верифицирует до 15 токенов за один проход по весам. Тот же результат, в 2.5 раза быстрее.",
      spec_run: "Запустить тест спекуляции",

      act_thermal: "ТЕМПЕРАТУРА",
      act_swap: "ДИСКОВЫЙ СВОП",
      act_ssd: "SSD KV СНИМКИ",
      act_context: "РАССЧИТАННЫЙ КОНТЕКСТ",

      integ_head: "Совместимость с OpenAI API",
      integ_sub_desc: "AI-llm-Engine работает на http://127.0.0.1:8000/v1. Готов к работе с Continue, Kilo Code, OpenCode, Claude Code и Cursor с автоматической защитой от переполнения токенов.",
      copy_config: "Скопировать конфиг",
      copy_url: "Скопировать URL сервера",

      gallery_kicker: "// СКРИНШОТЫ ПРИЛОЖЕНИЯ",
      gallery_headline: "Создано для macOS. Внутри и снаружи.",
      gallery_sub: "Скриншоты LLMBar в высоком разрешении на macOS Sonoma и Sequoia. Стиль Cyberpunk HUD, детальная телеметрия памяти и управление моделями.",
      shot_1_badge: "ВКЛАДКА ДВИЖОК · 35 TOK/S",
      shot_1_title: "Телеметрия ядра и скорость генерации",
      shot_1_desc: "Скорость генерации (34.8 ток/сек), префилл (119.5 ток/сек), сторож памяти и верификация Адамара за один шаг.",
      shot_2_badge: "ВКЛАДКА МОНИТОР · СТОРОЖ RAM",
      shot_2_title: "Аппаратный мониторинг и объединенная память",
      shot_2_desc: "Детализация в стиле Activity Monitor (веса, SSM, процесс, запас), гарантия 0 байт свопа и термоконтроль.",
      shot_3_badge: "ВКЛАДКА МОДЕЛИ · HUGGING FACE",
      shot_3_title: "Менеджер моделей Hugging Face",
      shot_3_desc: "Поиск, скачивание и переключение моделей в 1 клик. Проверка совместимости с объемом RAM и автоперезапуск.",
      shot_4_badge: "LLM EDITOR · ЧАТ",
      shot_4_title: "Агент, который спрашивает перед правкой",
      shot_4_desc: "Чат-ориентированный вид с режимами Manual, Edit, Plan и Auto: diff каждого изменения виден, и ничего не записывается без вашего подтверждения.",
      shot_5_badge: "LLM EDITOR · РЕДАКТОР",
      shot_5_title: "Нативный редактор рядом с ассистентом",
      shot_5_desc: "Дерево файлов, подсветка, проблемы, терминал и git с откатом по файлам — в небольшом Swift-приложении, которое работает только пока вы им пользуетесь.",

      arch_kicker: "// АРХИТЕКТУРА КРЕМНИЯ",
      arch_headline: "Почему это работает так быстро на Apple Silicon.",
      arch_sub: "Создано с нуля с полным использованием объединенной памяти Mac и компиляции Metal.",
      feat1_title: "Одно преобразование Адамара на общую размерность",
      feat1_desc: "Бандл содержит один вектор знаков на ширину входа. Проекции q|k|v, gate|up и qkv|z выполняются как единый спаренный квантованный matmul после одного преобразования вместо повторных на каждый слой.",
      feat2_title: "Промпт-спекуляция контекста",
      feat2_desc: "Генерирует продолжения n-грамм при правках кода и верифицирует до 15 токенов за шаг. Проверка 15 токенов стоит около 5 шагов обычного декодирования. Автоматически отключается на обычном тексте.",
      feat3_title: "Закрепленные (wired) веса",
      feat3_desc: "Лимит проводной памяти удерживает веса модели в физической RAM. macOS запрещено сжимать или выгружать их на диск, исключая зависания системы.",
      feat4_title: "Гибридный кэш префиксов на SSD",
      feat4_desc: "Состояние SSM и KV-кэш сохраняются на NVMe SSD по завершении запроса. Повторные запросы агентных чатов восстанавливаются мгновенно без повторного префилла.",
      feat5_title: "Адаптивный сторож RAM",
      feat5_desc: "Динамически вычисляет безопасный размер контекста. Уменьшает чанки префилла (1024 → 256 → 128) и удерживает память ниже 88%, предотвращая блокировку других приложений.",
      feat6_title: "15-минутный авто-сон",
      feat6_desc: "После 15 минут простоя движок выгружает веса и возвращает ~7.6 GB памяти macOS. Просыпается всего за 2 секунды при новом запросе.",
      feat7_title: "Тернарные 1.75 бит на одном Metal-ядре",
      feat7_desc: "Бандлы PTQ1_0, TQ1_G128 и JANG packed хранятся в памяти упакованными и считаются Metal-ядром с таблицами. Модель на 27B помещается в ~6 ГБ; остальное из MLX идёт через необязательный бэкенд mlx-lm.",
      feat8_title: "Нативный редактор кода — по требованию",
      feat8_desc: "Отдельное Swift-приложение, которое не загружается, пока вы его не открыли: чат-вид и вид редактора, агент с режимами Manual / Edit / Plan / Auto и управлением глубиной рассуждений, запуск и остановка модели из панели, git с откатом по файлам, GitHub, терминал, проблемы, панель предпросмотра, диктовка и кольцо контекста, которое сжимает чат. Тёмная и светлая темы.",
      feat9_title: "Необязательные модули — выключены по умолчанию",
      feat9_desc: "ModelScope как второй источник моделей, картинки прямо в чате через mflux, поиск в интернете, темы и сниппеты Open VSX, правки активаций, другой OpenAI-совместимый сервер как модель ассистента, настройки прокси и CA. Ничего не работает, пока вы не попросите.",
      feat10_title: "GGUF на собственном декодере движка",
      feat10_desc: "Файлы GGUF семейств Llama, Qwen2 и Qwen3 при загрузке перепаковываются в квантизацию MLX и исполняются самим движком — тот же сервер, кэш промпта и спекуляция, а логиты совпадают с llama.cpp. Всё остальное идёт через llama.cpp, который скачивается только по необходимости.",

      perf_kicker: "// ВЕРИФИЦИРОВАННЫЕ ЗАМЕРЫ",
      perf_headline: "Реальные цифры без прикрас.",
      perf_sub: "Замерено на безвентиляторном MacBook Air M5 с 16 GB памяти на модели dealignai/Bonsai-2-27B-CRACK-Ternary-JANG (27B, ~7.6 GB). Чередующиеся запуски с одинаковыми промптами.",
      perf_metric: "Метрика",
      perf_diff: "Разница",
      p_decode: "Скорость декода (256 токенов)",
      p_decode_sub: "Обычное авторегрессионное декодирование",
      p_spec: "Правки кода (Спекуляция)",
      p_spec_sub: "Совпадения n-грамм по промпту",
      p_prefill: "Скорость префилла (2.3K промпт)",
      p_prefill_sub: "Скорость обработки входящего контекста",
      p_cold: "Холодный старт до первого токена",
      p_cold_sub: "Загрузка, компиляция и ответ",
      p_overhead: "Накладные расходы памяти процесса",
      p_overhead_sub: "Память поверх самих весов модели",
      p_size: "Размер установки",
      p_size_sub: "Размер приложения и окружения",
      p_tokens: "Большой max_tokens (например, 32K)",
      p_tokens_sub: "Запросы от Kilo Code / Continue",

      mod_kicker: "// ВЫБОР ЖЕЛЕЗА",
      mod_headline: "Выберите модель под ваш Mac.",
      mod_sub: "Укажите объем объединенной памяти вашего Mac, чтобы увидеть подходящие модели и скорость.",
      ram_label: "Память Mac:",
      mod_best_16: "Рекомендовано для 16GB",
      mod_pro_tier: "Требует от 24GB",
      mod_entry: "Базовый уровень",
      mod_bonsai_desc: "27B гибридная dense/SSM модель. Тернарное квантование с поворотом Адамара. Отлично работает в 7.6 GB RAM.",
      mod_35b_desc: "MoE архитектура с квантованием Адамара. Максимальная мощность рассуждений для агентного программирования.",
      mod_4b_desc: "Сверхлегкая модель для быстрых ответов и Mac с 8 GB RAM. Загружается быстрее чем за 1 секунду.",

      inst_kicker: "// РАЗВЕРТЫВАНИЕ",
      inst_headline: "Готово за пару минут.",
      inst_sub: "Скачайте DMG-установщик или запустите через консольный CLI.",
      inst_rec: "РЕКОМЕНДУЕТСЯ",
      inst_dmg_title: "Установщик DMG для macOS",
      inst_dmg_desc: "Простой установщик: перетащите LLMBar.app в Программы. Включает CLI и фоновое меню.",
      step1: "Скачайте LLMBar-1.0.dmg из GitHub Releases.",
      step2: "Перетащите LLMBar.app в папку Программы.",
      step3: "Запустите из меню или Launchpad. Готово!",
      btn_download_now: "Скачать из GitHub Releases",
      inst_cli_tag: "ДЛЯ РАЗРАБОТЧИКОВ",
      inst_cli_title: "Терминал и Python CLI",
      inst_cli_desc: "Для тех, кто предпочитает командную строку, автоматизацию и изолированные виртуальные окружения.",
      copy_cli: "Скопировать команду для терминала",

      faq_kicker: "// БАЗА ЗНАНИЙ",
      faq_headline: "Часто задаваемые вопросы",
      faq_q1: "Почему AI-llm-Engine быстрее Ollama или vMLX?",
      faq_a1: "AI-llm-Engine применяет векторы знаков Адамара один раз на общую размерность входа, а не на каждую проекцию, компилируя их в спаренные matmul Metal. Кроме того, спекулятивное декодирование по контексту выдает 27–35 токенов/сек, а кэш компиляции ускоряет старт с 19 до 4 секунд.",
      faq_q2: "Действительно ли MacBook Air 16 GB может запустить 27B модель?",
      faq_a2: "Да! Благодаря тернарному квантованию Адамара (JANG / Prism) веса 27B занимают всего ~7.6 GB. Накладные расходы процесса ограничены 0.66 GB, а сторож RAM удерживает общую память системы ниже 88%, исключая своп на SSD и подвисания.",
      faq_q3: "Как подключить движок к Continue.dev или Cursor?",
      faq_a3: "Движок запускает стандартный OpenAI-совместимый HTTP-сервер на http://127.0.0.1:8000/v1. В Continue.dev укажите provider: 'openai', apiBase на этот адрес. Любой клиент OpenAI работает из коробки.",
      faq_q4: "Каковы системные требования macOS?",
      faq_a4: "Любой Mac на чипах Apple Silicon (M1, M2, M3, M4 или M5) под управлением macOS 14.0 (Sonoma) или macOS 15.0+ (Sequoia). Работает от 8 GB RAM (для 4B моделей) до 16, 24, 36, 64, 128 GB+.",

      cta_title: "Локальный интеллект на скорости кремния.",
      cta_sub: "Скачайте LLMBar.app прямо сейчас. 100% на устройстве, приватно и бесплатно под лицензией Apache 2.0.",
      cta_download: "Скачать из GitHub Releases",
      cta_github: "Репозиторий на GitHub"
    }
  };

  // State
  let currentLang = localStorage.getItem('apple_lang') || 'en';
  let currentTheme = localStorage.getItem('llm_theme') || 'cyberpunk-dark';

  // Apply Theme
  function applyTheme(theme) {
    currentTheme = theme;
    localStorage.setItem('llm_theme', theme);
    document.documentElement.setAttribute('data-theme', theme);

    const themeIcon = document.getElementById('theme-icon');
    const themeLabel = document.getElementById('theme-label');

    if (theme === 'apple-light') {
      if (themeIcon) themeIcon.textContent = '☼';
      if (themeLabel) themeLabel.textContent = currentLang === 'ru' ? 'Светлое стекло' : 'Light Glass';
    } else {
      if (themeIcon) themeIcon.textContent = '⚡';
      if (themeLabel) themeLabel.textContent = currentLang === 'ru' ? 'Темная HUD' : 'Dark HUD';
    }
  }

  // Toggle Theme
  document.getElementById('theme-toggle')?.addEventListener('click', () => {
    const nextTheme = currentTheme === 'cyberpunk-dark' ? 'apple-light' : 'cyberpunk-dark';
    applyTheme(nextTheme);
  });

  // Apply Language
  function applyLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('apple_lang', lang);

    document.querySelectorAll('.lang-pill').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === lang);
    });

    const dict = translations[lang] || translations.en;
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (dict[key]) {
        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
          el.placeholder = dict[key];
        } else {
          el.innerHTML = dict[key];
        }
      }
    });

    document.documentElement.lang = lang;
    applyTheme(currentTheme); // refresh theme label translation
  }

  // Language button event listeners
  document.getElementById('lang-en')?.addEventListener('click', () => applyLanguage('en'));
  document.getElementById('lang-ru')?.addEventListener('click', () => applyLanguage('ru'));

  // Mobile nav burger — the whole menu collapses behind it ≤1024px (see styles.css)
  const navBurger = document.getElementById('nav-burger');
  const navEl = document.querySelector('.apple-nav');
  const setNavOpen = (open) => {
    navEl?.classList.toggle('nav-open', open);
    navBurger?.setAttribute('aria-expanded', open ? 'true' : 'false');
    document.body.classList.toggle('nav-scroll-lock', open);
  };
  navBurger?.addEventListener('click', () => setNavOpen(!navEl.classList.contains('nav-open')));
  // tapping an actual link closes the drawer; theme/language toggles stay open
  document.querySelectorAll('.nav-center a, .nav-right a').forEach(el => {
    el.addEventListener('click', () => setNavOpen(false));
  });
  window.addEventListener('resize', () => {
    if (window.innerWidth > 1024) setNavOpen(false);
  });

  // ==========================================
  // 2. Tab Navigation (macOS Segmented Bar)
  // ==========================================
  const tabButtons = document.querySelectorAll('.seg-btn');
  const tabPanes = document.querySelectorAll('.window-pane');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');

      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      tabPanes.forEach(pane => {
        pane.classList.remove('active');
        if (pane.id === `pane-${targetTab}`) {
          pane.classList.add('active');
        }
      });
    });
  });

  // Top Tray Pill Click -> smooth scroll to app demo
  document.getElementById('tray-pill')?.addEventListener('click', () => {
    document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' });
    const engineTabBtn = document.getElementById('tab-engine');
    if (engineTabBtn) engineTabBtn.click();
  });

  // Buttons inside app window
  document.getElementById('btn-demo-chat')?.addEventListener('click', () => {
    document.getElementById('tab-terminal')?.click();
  });

  document.getElementById('btn-copy-api')?.addEventListener('click', (e) => {
    copyToClipboard('http://127.0.0.1:8000/v1', e.currentTarget);
  });

  const sleepBtn = document.getElementById('btn-sleep-test');
  if (sleepBtn) {
    let sleeping = false;
    sleepBtn.addEventListener('click', () => {
      sleeping = !sleeping;
      const statusBadge = document.querySelector('.window-engine-badge span:last-child');
      const pulseDot = document.querySelector('.engine-pulse');
      if (sleeping) {
        sleepBtn.innerHTML = '<span>⚡</span> <span>Wake Engine</span>';
        if (statusBadge) statusBadge.textContent = 'SLEEPING (0 GB)';
        if (pulseDot) pulseDot.style.background = '#86868b';
      } else {
        sleepBtn.innerHTML = '<span>🌙</span> <span>Sleep Engine</span>';
        if (statusBadge) statusBadge.textContent = currentLang === 'ru' ? 'АКТИВЕН' : 'ACTIVE';
        if (pulseDot) pulseDot.style.background = 'var(--hud-phosphor)';
      }
    });
  }

  // ==========================================
  // 3. Apple Silicon Sparkline Telemetry
  // ==========================================
  const canvas = document.getElementById('sparklineCanvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    let width = canvas.width;
    let height = canvas.height;

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      width = rect.width;
      height = rect.height;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const points = [
      8.2, 8.4, 8.1, 9.2, 28.5, 34.2, 35.1, 32.8, 34.9, 35.2,
      27.8, 35.0, 34.7, 33.9, 35.4, 34.8, 35.1, 34.6, 35.2, 34.9
    ];

    let step = 0;
    const decodeValEl = document.getElementById('live-decode-val');

    function drawSparkline() {
      ctx.clearRect(0, 0, width, height);

      // Baseline grid
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height * 0.25); ctx.lineTo(width, height * 0.25);
      ctx.moveTo(0, height * 0.75); ctx.lineTo(width, height * 0.75);
      ctx.stroke();

      // Smooth Curve
      ctx.beginPath();
      const dx = width / (points.length - 1);
      const minVal = 0;
      const maxVal = 40;

      for (let i = 0; i < points.length; i++) {
        const x = i * dx;
        const norm = (points[i] - minVal) / (maxVal - minVal);
        const y = height - (norm * (height - 18) + 9);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          const prevX = (i - 1) * dx;
          const prevNorm = (points[i - 1] - minVal) / (maxVal - minVal);
          const prevY = height - (prevNorm * (height - 18) + 9);
          const cx = (prevX + x) / 2;
          ctx.bezierCurveTo(cx, prevY, cx, y, x, y);
        }
      }

      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2.4;
      ctx.lineCap = 'round';
      ctx.stroke();

      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      const fillGrad = ctx.createLinearGradient(0, 0, 0, height);
      fillGrad.addColorStop(0, 'rgba(0, 229, 255, 0.25)');
      fillGrad.addColorStop(1, 'rgba(0, 229, 255, 0.0)');
      ctx.fillStyle = fillGrad;
      ctx.fill();

      // Glowing dot
      const lastX = width;
      const lastNorm = (points[points.length - 1] - minVal) / (maxVal - minVal);
      const lastY = height - (lastNorm * (height - 18) + 9);

      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#4cf2c2';
      ctx.shadowColor = '#4cf2c2';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;

      step++;
      if (step % 25 === 0) {
        const newVal = 33.5 + Math.random() * 2.2;
        points.shift();
        points.push(newVal);
        if (decodeValEl) {
          decodeValEl.textContent = newVal.toFixed(1);
        }
      }

      requestAnimationFrame(drawSparkline);
    }

    requestAnimationFrame(drawSparkline);
  }

  // ==========================================
  // 4. Speculation Demo Streamer
  // ==========================================
  const runSpecBtn = document.getElementById('btn-run-spec');
  const streamerOutput = document.getElementById('streamer-output');
  const streamerStatus = document.getElementById('streamer-status-tag');
  const speedMeter = document.getElementById('spec-speed-meter');

  if (runSpecBtn && streamerOutput) {
    const codeLines = [
      '<span class="tk-comment"># Speculative prompt-lookup: checking 15-gram matches against context</span>',
      '<span class="tk-keyword">def</span> <span class="tk-function">apply_fused_quantized_hadamard</span>(x, weights, signs):',
      '    <span class="tk-string">"""Single transform per shared input width with exact verification."""</span>',
      '    <span class="tk-comment"># In vMLX: 3 Hadamard transforms. In AI-llm-Engine: 1 Hadamard transform</span>',
      '    h = mx.compile(hadamard_fast_walsh)(x, signs)',
      '    q_out = mx.quantized_matmul(h, weights, transpose=True)',
      '    <span class="tk-keyword">return</span> q_out',
      '',
      '<span class="tk-comment"># Speculation verified 15 drafted tokens in 1 weight pass [Hit rate: 94%]</span>',
      '<span class="tk-keyword">print</span>(<span class="tk-string">"Decode complete: 35.4 tok/s sustained"</span>)'
    ];

    let isRunning = false;

    runSpecBtn.addEventListener('click', () => {
      if (isRunning) return;
      isRunning = true;
      runSpecBtn.disabled = true;
      if (streamerStatus) streamerStatus.textContent = 'STREAMING (35 tok/s)';
      if (speedMeter) speedMeter.textContent = '35.4 tok/s (×2.5)';
      streamerOutput.innerHTML = '';

      let lineIdx = 0;
      const interval = setInterval(() => {
        if (lineIdx < codeLines.length) {
          streamerOutput.innerHTML += codeLines[lineIdx] + '\n';
          lineIdx++;
        } else {
          clearInterval(interval);
          isRunning = false;
          runSpecBtn.disabled = false;
          if (streamerStatus) streamerStatus.textContent = 'VERIFIED ✓';
        }
      }, 95);
    });
  }

  // ==========================================
  // 5. RAM Calculator & Model Selector
  // ==========================================
  const ramPills = document.querySelectorAll('.ram-pill');
  const modelCards = document.querySelectorAll('.apple-model-card');

  ramPills.forEach(pill => {
    pill.addEventListener('click', () => {
      ramPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const selectedRam = parseInt(pill.getAttribute('data-ram'), 10);

      modelCards.forEach(card => {
        const minRam = parseInt(card.getAttribute('data-min-ram'), 10);
        if (minRam > selectedRam) {
          card.classList.add('dimmed');
        } else {
          card.classList.remove('dimmed');
        }
      });
    });
  });

  // ==========================================
  // 6. Copy to Clipboard Functionality
  // ==========================================
  function copyToClipboard(text, triggerEl) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => showCopiedFeedback(triggerEl));
    } else {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      showCopiedFeedback(triggerEl);
    }
  }

  function showCopiedFeedback(el) {
    if (!el) return;
    const originalText = el.innerHTML;
    el.innerHTML = `<span style="color: var(--hud-phosphor)">✓ ${currentLang === 'ru' ? 'Скопировано' : 'Copied'}</span>`;
    setTimeout(() => {
      el.innerHTML = originalText;
    }, 2000);
  }

  const quickCopyBtn = document.getElementById('copy-quick-cmd');
  const quickCmdEl = document.getElementById('quick-cmd');
  if (quickCopyBtn && quickCmdEl) {
    quickCopyBtn.addEventListener('click', () => {
      copyToClipboard(quickCmdEl.textContent, quickCopyBtn);
    });
  }

  document.querySelectorAll('.copy-snippet-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const snippet = btn.getAttribute('data-snippet');
      if (snippet) {
        copyToClipboard(snippet, btn);
      }
    });
  });

  // Initialize
  applyTheme(currentTheme);
  applyLanguage(currentLang);
})();
