# Graphy Codebase Analysis

## Project: /Users/dbk/Desktop/agentic-graph-RAG

### File Structure Summary
- Total Files: 321
- Total Directories: 67
- File Extensions: .md: 23, .txt: 11, .py: 188, .json: 56, .yml: 1, .db: 6, .gguf: 1, .pt: 5, .pth: 4, .csv: 1, .whl: 1, .safetensors: 13, .sig: 2, .jinja: 2, .h5: 1, .msgpack: 1, .html: 1, .wal: 1

### Directory Tree
```
  ├── sensors
   │  ├── audio_module
   │  │  ├── config.py
   │  │  ├── sinks
   │  │  │  ├── embedding_store.py
   │  │  │  └── jsonl_sink.py
   │  │  ├── segment
   │  │  │  ├── ringbuffer.py
   │  │  │  └── segmenter.py
   │  │  ├── runtime
   │  │  │  ├── run_cli.py
   │  │  │  └── pipeline.py
   │  │  ├── __init__.py
   │  │  ├── schema
   │  │  │  └── audioseg.py
   │  │  ├── processors
   │  │  │  ├── whisper_asr.py
   │  │  │  ├── efficientat.py
   │  │  │  ├── policy.py
   │  │  │  ├── efficientat_parts
   │  │  │  │  ├── __init__.py
   │  │  │  │  └── panns_processor.py
   │  │  │  ├── vad.py
   │  │  │  └── quality.py
   │  │  ├── sources
   │  │  │  ├── file_source.py
   │  │  │  ├── push_source.py
   │  │  │  ├── mic_source.py
   │  │  │  └── base.py
   │  │  └── patched_download.py
   │  ├── text_module
   │  │  ├── config.py
   │  │  ├── sinks
   │  │  │  ├── embedding_store.py
   │  │  │  └── jsonl_sink.py
   │  │  ├── runtime
   │  │  │  ├── run_cli.py
   │  │  │  └── pipeline.py
   │  │  ├── __init__.py
   │  │  ├── schema
   │  │  │  └── textseg.py
   │  │  ├── processors
   │  │  │  ├── bert_embedder.py
   │  │  │  ├── language.py
   │  │  │  ├── normalization.py
   │  │  │  └── command_parser.py
   │  │  └── sources
   │  │    ├── file_source.py
   │  │    ├── push_source.py
   │  │    └── base.py
   │  └── vision_module
   │    ├── config.py
   │    ├── sinks
   │     │  ├── embedding_store.py
   │     │  ├── snapshot_store.py
   │     │  └── jsonl_sink.py
   │    ├── runtime
   │     │  ├── run_cli.py
   │     │  ├── pipeline.py
   │     │  └── pipeline_parts
   │     │    ├── builders_annotations_and_recovery.py
   │     │    ├── __init__.py
   │     │    ├── lifecycle_and_detection.py
   │     │    └── tracking_policy_and_outputs.py
   │    ├── sampling
   │     │  └── frame_sampler.py
   │    ├── __init__.py
   │    ├── schema
   │     │  └── visionseg.py
   │    ├── processors
   │     │  ├── yolo_wrappers.py
   │     │  ├── policy.py
   │     │  ├── features.py
   │     │  ├── projector.py
   │     │  ├── tracker.py
   │     │  ├── quality.py
   │     │  └── scene_classifier.py
   │    ├── sources
   │     │  ├── rtsp_source.py
   │     │  ├── file_source.py
   │     │  ├── push_source.py
   │     │  ├── camera_source.py
   │     │  └── base.py
   │    └── cv2_compat.py
  ├── ingestion
   │  ├── entity_memory.py
   │  ├── context_relation_extractor.py
   │  ├── relation_normalizer.py
   │  ├── context_builder.py
   │  ├── chunk_ingestion_pipeline.py
   │  ├── semantic_graph_builder.py
   │  ├── datasets
   │  │  ├── relations.py
   │  │  └── entities.py
   │  ├── entity_canonicalizer.py
   │  ├── ner_extractor.py
   │  ├── relation_extractor.py
   │  ├── relation_pipeline.py
   │  ├── entity_linker.py
   │  ├── relation_ontology.py
   │  ├── llm_relation_extractor.py
   │  ├── EntityExtractionPipeline.py
   │  ├── embedding_generator.py
   │  ├── graph_builder.py
   │  └── docling_processor.py
  ├── llm
   │  ├── prompt_understanding.py
   │  ├── tool_registry.py
   │  ├── answer_generation.py
   │  ├── reasoning
   │  │  ├── __init__.py
   │  │  └── llm_reasoning.py
   │  ├── prompt_understanding copy.py
   │  ├── mlx_granite.py
   │  ├── base_llm.py
   │  └── granite_client
   │    ├── mlx_client.py
   │    ├── config.py
   │    ├── Ollama_client.py
   │    ├── base_client.py
   │    ├── __init__.py
   │    ├── prompts.py
   │    └── llamacpp_client.py
  ├── test_kuzu_kg
  ├── core
   │  ├── config.py
   │  ├── constants.py
   │  ├── model_manager.py
   │  ├── logger.py
   │  ├── model_wrapper.py
   │  ├── exceptions.py
   │  ├── model_gate.py
   │  ├── metrics_monitor.py
   │  └── memory_monitor.py
  ├── test_kuzu_db
  ├── IMPLEMENTATION_SUMMARY.md
  ├── test_kuzu_direct
  ├── external_sources
   │  ├── web_search_adapter.py
   │  ├── __init__.py
   │  └── data_collector
   │    ├── collectors
   │     │  ├── social.py
   │     │  ├── __init__.py
   │     │  ├── financial.py
   │     │  ├── scientific.py
   │     │  └── knowledge.py
   │    ├── models.py
   │    ├── timeout.py
   │    ├── __init__.py
   │    ├── collected_data.json
   │    ├── # environment.yml
   │    ├── example.py
   │    ├── helper.py
   │    ├── exceptions.py
   │    └── manager.py
  ├── tests
   │  ├── test_system_integration_simple.py
   │  ├── test_end_to_end_integration.py
   │  ├── test_main_adaptive_production.py
   │  ├── test_end_to_end_complete.py
   │  ├── test_kuzu_integration.py
   │  ├── test_adaptive_orchestrator.py
   │  ├── test_phi4_integration.py
   │  ├── test_memory_management.py
   │  ├── test_query_classifier_simple.py
   │  ├── test_all_skills.py
   │  └── test_mdskills_mcp.py
  ├── test_db
   │  └── graph.db
  ├── agents
   │  ├── ai_model_skills.py
   │  ├── advanced_skills.py
   │  ├── sensor_integration_system.py
   │  ├── enhanced_knowledge_refresh_agent.py
   │  ├── reasoner_agent.py
   │  ├── tool_registry.py
   │  ├── adaptive_orchestrator.py
   │  ├── __init__.py
   │  ├── validator_agent.py
   │  ├── improved_knowledge_refresh_agent.py
   │  ├── skill_executor.py
   │  ├── query_classifier.py
   │  ├── knowledge_refresh_agent.py
   │  ├── orchestrator.py
   │  ├── retriever_agent.py
   │  ├── document_processing_skills.py
   │  ├── sensor_skills.py
   │  ├── specialized_skills.py
   │  └── state.py
  ├── mcp
   │  ├── tool_handlers.py
   │  ├── mdskills_mcp_tools.py
   │  ├── resource_handlers.py
   │  └── mcp_server.py
  ├── test_kuzu_simple
  ├── models
   │  ├── phi3_mini
   │  │  ├── CODE_OF_CONDUCT.md
   │  │  ├── data_summary_card.md
   │  │  ├── added_tokens.json
   │  │  ├── tokenizer_config.json
   │  │  ├── special_tokens_map.json
   │  │  ├── model-00001-of-00002.safetensors
   │  │  ├── config.json
   │  │  ├── tokenizer.json
   │  │  ├── modeling_phi3.py
   │  │  ├── generation_config.json
   │  │  ├── README.md
   │  │  ├── merges.txt
   │  │  ├── model-00002-of-00002.safetensors
   │  │  ├── vocab.json
   │  │  ├── configuration_phi3.py
   │  │  ├── sample_finetune.py
   │  │  ├── LICENSE.txt
   │  │  ├── model.safetensors.index.json
   │  │  ├── NOTICE.md
   │  │  └── SECURITY.md
   │  ├── Phi-3-mini-4k-instruct-q4-2.gguf
   │  ├── yolo11xseg.pt
   │  ├── granite4-7b
   │  │  ├── model-00002-of-00003.safetensors
   │  │  ├── tokenizer_config.json
   │  │  ├── special_tokens_map.json
   │  │  ├── model.sig
   │  │  ├── config.json
   │  │  ├── model-00003-of-00003.safetensors
   │  │  ├── tokenizer.json
   │  │  ├── generation_config.json
   │  │  ├── merges.txt
   │  │  ├── chat_template.jinja
   │  │  ├── vocab.json
   │  │  ├── model.safetensors.index.json
   │  │  └── model-00001-of-00003.safetensors
   │  ├── all-mpnet-base-v2
   │  │  ├── model.safetensors
   │  │  ├── 0_Transformer
   │  │  ├── 1_Pooling
   │  │  │  └── config.json
   │  │  ├── tokenizer_config.json
   │  │  ├── special_tokens_map.json
   │  │  ├── config.json
   │  │  ├── tokenizer.json
   │  │  ├── vocab.txt
   │  │  └── modules.json
   │  ├── Cnn10.pth
   │  ├── yolo11x.pt
   │  ├── Cnn14.pth
   │  ├── class_labels_indices.csv
   │  ├── labels.txt
   │  ├── granite4_3b
   │  │  ├── tokenizer_config.json
   │  │  ├── special_tokens_map.json
   │  │  ├── model-00001-of-00002.safetensors
   │  │  ├── model.sig
   │  │  ├── config.json
   │  │  ├── tokenizer.json
   │  │  ├── generation_config.json
   │  │  ├── README.md
   │  │  ├── merges.txt
   │  │  ├── chat_template.jinja
   │  │  ├── model-00002-of-00002.safetensors
   │  │  ├── vocab.json
   │  │  └── model.safetensors.index.json
   │  ├── image_model.pth
   │  ├── whisper-medium
   │  │  ├── model.safetensors
   │  │  ├── added_tokens.json
   │  │  ├── tokenizer_config.json
   │  │  ├── special_tokens_map.json
   │  │  ├── config.json
   │  │  ├── tokenizer.json
   │  │  ├── generation_config.json
   │  │  ├── normalizer.json
   │  │  ├── README.md
   │  │  ├── merges.txt
   │  │  ├── vocab.json
   │  │  ├── tf_model.h5
   │  │  ├── medium.pt
   │  │  ├── flax_model.msgpack
   │  │  └── preprocessor_config.json
   │  ├── gpt2-small
   │  │  ├── model.safetensors
   │  │  ├── tokenizer_config.json
   │  │  ├── config.json
   │  │  ├── tokenizer.json
   │  │  ├── merges.txt
   │  │  └── vocab.json
   │  ├── yolo11xpos.pt
   │  ├── en_core_web_trf-any-py3-none-any.whl
   │  ├── bert-base-cased
   │  │  ├── model.safetensors
   │  │  ├── tokenizer_config.json
   │  │  ├── config.json
   │  │  ├── tokenizer.json
   │  │  ├── tokenizer3.json
   │  │  ├── vocab.txt
   │  │  ├── config3.json
   │  │  └── tokenizer_config3.json
   │  ├── efficientat.pt
   │  ├── gpt2_larg.safetensors
   │  ├── Qwen3-Embedding-0.6B
   │  │  ├── model.safetensors
   │  │  ├── tokenizer_config.json
   │  │  ├── config.json
   │  │  ├── config_sentence_transformers.json
   │  │  ├── tokenizer.json
   │  │  ├── generation_config.json
   │  │  ├── README.md
   │  │  ├── merges.txt
   │  │  ├── vocab.json
   │  │  └── modules.json
   │  └── wav2vec2_fairseq_base_ls960_asr_ls960.pth
  ├── docs
   │  ├── SKILL_TOOL_CALL_SYSTEM.md
   │  ├── PHI4_IMPROVEMENTS.md
   │  ├── ENHANCED_KNOWLEDGE_REFRESH_GUIDE.md
   │  ├── COMPLETE_FLOW_ANALYSIS.md
   │  ├── ADAPTIVE_MAIN_GUIDE.md
   │  ├── UI_VISUALIZATION_GUIDE.md
   │  ├── MDSKILLS_MCP_GUIDE.md
   │  ├── ADVANCED_SKILLS.md
   │  ├── KNOWLEDGE_REFRESH_IMPROVEMENT.md
   │  ├── MEMORY_MANAGEMENT.md
   │  ├── DYNAMIC_FLOW_CONTROL.md
   │  ├── PHI4_INTEGRATION_ANALYSIS.md
   │  └── KUZU_SETUP_GUIDE.md
  ├── test_db_async
   │  └── graph.db
  ├── knowledg_graph
   │  ├── kuzudb_package
   │  │  ├── async_manager.py
   │  │  ├── models.py
   │  │  ├── client.py
   │  │  ├── __init__.py
   │  │  ├── helper.py
   │  │  ├── exceptions.py
   │  │  ├── demo_db
   │  │  │  └── kuzu.db
   │  │  └── manager.py
   │  ├── kuzu_async.py
   │  ├── graph_traversal.py
   │  ├── subgraph_extractor.py
   │  ├── kuzu_client.py
   │  ├── kuzu_wrapper.py
   │  ├── example.py
   │  ├── cypher_executor.py
   │  └── schema_manager.py
  ├── vector_store
   │  ├── weaviate_client.py
   │  ├── hybrid_search.py
   │  ├── collection_manager.py
   │  └── vector_search.py
  ├── requirements_ui.txt
  ├── README.md
  ├── search_engine
   │  ├── opensearch_client.py
   │  ├── bm25_search.py
   │  └── index_manager.py
  ├── logs
   │  └── metrics
   │    └── metrics_20260608_231542.json
  ├── retrieval
   │  ├── fusion_retriever.py
   │  ├── context_builder.py
   │  └── gap_detector.py
  ├── main_adaptive.py
  ├── main_adaptive_enhanced.py
  ├── workflow
   │  └── agentic_graph_rag.py
  ├── configs
   │  └── main_config.py
  ├── api
   │  ├── visualization_api.py
   │  ├── dashboard.py
   │  └── static
   │    └── index.html
  ├── demo_db
   │  └── graph.db
  ├── embeding_test.py
  ├── main.py
  ├── data
   │  ├── kuzu_db
   │  │  ├── graph.db.wal
   │  │  └── graph.db
   │  └── kuzu_db.old-schema-backup
   │    └── graph.db
  └── routes
```

### File List
- /Users/dbk/Desktop/agentic-graph-RAG/test_kuzu_kg
- /Users/dbk/Desktop/agentic-graph-RAG/IMPLEMENTATION_SUMMARY.md
- /Users/dbk/Desktop/agentic-graph-RAG/test_kuzu_direct
- /Users/dbk/Desktop/agentic-graph-RAG/test_kuzu_simple
- /Users/dbk/Desktop/agentic-graph-RAG/requirements_ui.txt
- /Users/dbk/Desktop/agentic-graph-RAG/README.md
- /Users/dbk/Desktop/agentic-graph-RAG/main_adaptive.py
- /Users/dbk/Desktop/agentic-graph-RAG/main_adaptive_enhanced.py
- /Users/dbk/Desktop/agentic-graph-RAG/embeding_test.py
- /Users/dbk/Desktop/agentic-graph-RAG/main.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/config.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/__init__.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/patched_download.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/sinks/embedding_store.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/sinks/jsonl_sink.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/segment/ringbuffer.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/segment/segmenter.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/runtime/run_cli.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/runtime/pipeline.py
- /Users/dbk/Desktop/agentic-graph-RAG/sensors/audio_module/schema/audioseg.py

... and 301 more files
