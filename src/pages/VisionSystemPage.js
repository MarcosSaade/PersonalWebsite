import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

import PageTemplate from '../components/PageTemplate';
import visionImage from '../images/vision-system.png';
import demoGif from '../images/vision-system/demo.gif';
import architectureDiagram from '../images/vision-system/architecture.png';
import performanceComparison from '../images/vision-system/yolo11_performance_comparison.png';
import rubikpiPhoto from '../images/vision-system/rubikpi.jpg';

export default function VisionSystemPage() {
  return (
    <PageTemplate title="Building an AI Vision System: From Natural Language Queries to Real-Time Edge Detection" image={visionImage}>
      {/* Introduction */}
      <p>
        I built an AI-powered vision system that transforms how users interact with computer vision data: users type natural language queries like "Show me all frames with people from yesterday," and the system automatically translates them into SQL, executes optimized database searches, and returns visual results with detection metadata.
      </p>
      <p>
        The system combines three core technologies: real-time object detection with YOLO11, Large Language Model integration for natural language understanding, and hardware-accelerated inference on edge devices. The result is a production-ready solution that processes video at 27 FPS while running entirely on-device with complete data privacy.
      </p>
      <p>
        The key technical challenge was achieving real-time performance on resource-constrained edge hardware. Initial CPU-based inference achieved only 3 FPS—unusable for real-time applications. By implementing INT8 quantization and leveraging Qualcomm's Neural Processing Unit (NPU), I achieved an <strong>8x speedup</strong> (3 FPS → 27 FPS) while simultaneously reducing operating temperatures by 0.27°C and maintaining detection accuracy.
      </p>
      <p>
        In this project, I had the chance of working with a some popular techniques in ML: model optimization and quantization, hardware acceleration, full-stack development, database optimization, and production system design. It's deployed on edge hardware, processes live video streams, and provides a natural language interface that makes sophisticated computer vision capabilities accessible to non-technical users.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={demoGif} alt="Natural language query system demo" style={{maxWidth: '100%', height: 'auto', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{marginTop: '10px', fontStyle: 'italic', color: '#666', fontSize: '0.9em'}}>
          Natural language queries in action (using images from COCO dataset instead of actual Rubik Pi camera for privacy).
        </p>
      </div>

      <h2>Problem Statement: Making Computer Vision Accessible</h2>
      <p>
        Traditional video anaysis systems require users to sit through hours of video to find specific events or patterns. Even with object detection, search interfaces are often limited to conventional filters that don't capture the richness of natural language queries.
      </p>
      <p>
        I designed a natural language interface that accepts queries in plain English:
      </p>
      <ul>
        <li>"How many people were detected today?"</li>
        <li>"Show me frames with cars from yesterday"</li>
        <li>"What are the top 5 most detected objects?"</li>
        <li>"Show me the hourly trend for apples today"</li>
      </ul>
      <p>
        The technical requirements were clear: translate natural language to SQL with high accuracy, ensure query safety and validation, optimize database performance for sub-second responses, and stream results progressively to minimize perceived latency. Success required expertise in LLM prompt engineering, database optimization, and real-time streaming architectures.
      </p>

      <h2>System Architecture: The Full Stack</h2>
      <p>
        The system consists of five main components working in concert:
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={architectureDiagram} alt="System architecture diagram" style={{maxWidth: '100%', height: 'auto', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{marginTop: '10px', fontStyle: 'italic', color: '#666', fontSize: '0.9em'}}>
          System architecture.
        </p>
      </div>

      <h3>1. Frontend: React with Real-Time Streaming</h3>
      <p>
        The React-based frontend implements Server-Sent Events (SSE) for token-by-token streaming from the LLM, reducing perceived latency by showing progressive updates as queries are generated and executed. Users receive immediate feedback rather than waiting for complete responses, significantly improving the user experience.
      </p>
      <p>
        The interface provides multiple interaction modes: natural language queries for non-technical users, conventional filter-based search, and provider switching between edge (Ollama) and cloud (Gemini) AI. Live video streams display detection overlays with bounding boxes, confidence scores, and class labels in real-time.
      </p>
      <p>
        Key implementation details include connection state management for SSE streams, graceful error recovery with automatic reconnection, and efficient rendering of detection visualizations without UI lag.
      </p>

      <h3>2. Backend: FastAPI with Async Processing</h3>
      <p>
        The FastAPI backend implements a fully asynchronous architecture enabling concurrent request handling and non-blocking I/O operations. This design supports multiple simultaneous users while maintaining responsive performance.
      </p>
      <p>
        Core responsibilities include:
      </p>
      <ul>
        <li><strong>LLM Orchestration:</strong> Prompt construction with schema context, few-shot examples, and safety constraints</li>
        <li><strong>SQL Validation Pipeline:</strong> Multi-layer validation including syntax checking, whitelist enforcement, limit verification, and retry logic for self-correction</li>
        <li><strong>Streaming Infrastructure:</strong> SSE-based token streaming with connection management and graceful degradation</li>
      </ul>
      <p>
        The async architecture was essential for achieving sub-second query responses while streaming token updates—synchronous processing would have blocked the entire pipeline during LLM inference.
      </p>

      <h3>3. LLM Integration: Dual-Mode AI with Provider Abstraction</h3>
      <p>
        I implemented a provider abstraction layer supporting both on-device and cloud inference, enabling users to choose between data privacy (local inference) and maximum performance (cloud API):
      </p>
      <ul>
        <li><strong>Ollama (Edge AI):</strong> Local inference with qwen3:1.7b—complete data privacy, zero latency to first token, no API costs</li>
        <li><strong>Google Gemini (Cloud AI):</strong> Higher accuracy and faster inference with access to larger context windows</li>
      </ul>
      <p>
        The abstraction layer implements a consistent interface across providers, enabling hot-swapping without code changes and simplifying future provider additions. Prompt engineering was critical for reliable SQL generation:
      </p>
      <ul>
        <li><strong>Schema Documentation:</strong> Complete table definitions with types, constraints, and semantic descriptions</li>
        <li><strong>Few-Shot Learning:</strong> 15 example queries demonstrating temporal expressions, aggregations, and JOINs</li>
        <li><strong>Contextual Information:</strong> Current timezone for temporal query resolution</li>
        <li><strong>Safety Constraints:</strong> Explicit rules for LIMIT clauses, read-only operations, and error handling</li>
      </ul>

      <h3>4. Detection Pipeline: Hardware-Accelerated YOLO11</h3>
      <p>
        Object detection runs continuously in the background, capturing frames from one or more connected cameras camera and processing them with YOLO11. Each detected object (from 80 COCO classes including person, car, dog, etc.) is logged to the database with:
      </p>
      <ul>
        <li>Timestamp (for temporal queries)</li>
        <li>Object class and confidence score</li>
        <li>Bounding box coordinates</li>
        <li>Frame reference and URI</li>
      </ul>

      <h3>5. Database: SQLite</h3>
      <p>
        I chose SQLite for its simplicity and efficiency. The schema is optimized for the types of queries users would naturally ask:
      </p>
      <ul>
        <li><strong>Temporal queries:</strong> Index on timestamp for "show me detections from yesterday"</li>
        <li><strong>Categorical queries:</strong> Index on object class for "how many cars were detected"</li>
        <li><strong>Frame retrieval:</strong> Efficient JOIN between detections and frames tables</li>
      </ul>
      <p>
        Proper indexing made queries 5-10x faster, turning potentially sluggish searches into snappy, responsive results.
      </p>

      <h2>Natural Language to SQL: Engineering for Reliability</h2>
      <p>
        Translating arbitrary natural language into production-grade SQL requires handling multiple challenges: schema understanding, temporal expression parsing ("yesterday", "last week"), SQL syntax correctness, query optimization, and security constraints. Naive implementations produce unreliable results—careful prompt engineering and validation are essential.
      </p>

      <h3>Prompt Engineering</h3>
      <p>
        SQL generation quality is entirely dependent on prompt quality. Through iterative refinement, I developed a prompt structure achieving ~95% first-attempt accuracy:
      </p>
      <ul>
        <li><strong>Schema Context:</strong> Complete DDL with semantic field descriptions enabling schema understanding</li>
        <li><strong>Few-Shot Examples:</strong> 15 diverse query patterns (aggregations, temporal filters, JOINs, subqueries)</li>
        <li><strong>Timezone Context:</strong> Dynamic timezone injection for accurate temporal query resolution</li>
        <li><strong>Constraint Specification:</strong> Explicit requirements for LIMIT clauses, read-only operations, and error handling</li>
      </ul>
      <p>
        The LLM returns structured JSON with three components:
      </p>
      <ul>
        <li><strong>sql:</strong> The generated query, validated before execution</li>
        <li><strong>explanation:</strong> Human-readable query explanation for transparency</li>
        <li><strong>visualization_type:</strong> Optimal display format (count, table, chart, frames) based on query semantics</li>
      </ul>

      <h3>SQL Validation</h3>
      <p>
        Allowing LLM-generated SQL requires rigorous validation. I implemented a four-layer security and correctness pipeline:
      </p>
      <ul>
        <li><strong>Syntax Validation:</strong> AST parsing to verify syntactic correctness before execution</li>
        <li><strong>Operation Whitelist:</strong> Strict allowlist permitting only SELECT statements—reject DELETE, UPDATE, DROP, CREATE</li>
        <li><strong>Resource Limits:</strong> Enforce LIMIT clauses on all queries to prevent resource exhaustion</li>
        <li><strong>Self-Correction Loop:</strong> On validation failure, feed error details back to the LLM for iterative correction (typically succeeds by attempt 2)</li>
      </ul>
      <p>
        The self-correction mechanism is particularly effective: the LLM receives not just "query failed" but the specific error message, enabling intelligent correction. This approach achieves ~98% success rate within two attempts.
      </p>

      <h3>Streaming for Low Latency</h3>
      <p>
        Even with fast LLMs, generating SQL can take several seconds. Rather than making users wait for a complete response, the system streams tokens as they're generated. Here's the flow:
      </p>
      <ol>
        <li>User submits a natural language query</li>
        <li>Backend sends "processing" status (frontend shows loading indicator)</li>
        <li>LLM starts generating—each token streams immediately to frontend</li>
        <li>User sees the SQL being built in real-time</li>
        <li>Once complete, SQL is validated and executed</li>
        <li>Results stream back with frame thumbnails and metadata</li>
      </ol>
      <p>
        This progressive enhancement makes the system feel faster and more responsive, even though total time may be unchanged.
      </p>

      <h2>Hardware Acceleration: Achieving Production-Ready Performance</h2>
      <p>
        Initial CPU-based inference achieved only 3 FPS (304ms per frame)—inadequate for real-time applications requiring 24-30 FPS. The computational bottleneck was clear: YOLO11 executes millions of multiply-accumulate operations per frame, and general-purpose CPUs lack the parallelism and specialized instructions needed for efficient neural network inference.
      </p>
      <p>
        Real-time performance (30 FPS) requires sub-33ms inference latency. CPU inference at 304ms was too slow.
      </p>

      <h3>Solution: NPU Acceleration with INT8 Quantization</h3>
      <p>
        I leveraged the Qualcomm Hexagon Tensor Processor (HTP)—a Neural Processing Unit in the QCS6490 SoC—designed specifically for neural network inference. NPUs deliver order-of-magnitude speedups through three architectural advantages:
      </p>
      <ul>
        <li><strong>Massive Parallelism:</strong> Thousands of multiply-accumulate (MAC) units executing simultaneously</li>
        <li><strong>Quantized Arithmetic:</strong> Hardware-optimized INT8/INT4 operations with 4x memory bandwidth reduction</li>
        <li><strong>On-Chip Memory:</strong> Minimized DRAM access, reducing both latency and power consumption</li>
      </ul>
      <p>
        Deploying YOLO11 to the NPU required model conversion, quantization, and runtime optimization.
      </p>

      <h3>Quantitative Results: 8x Performance Improvement</h3>
      <p>
        NPU acceleration delivered substantial, measurable improvements across all performance metrics:
      </p>
      <table style={{margin: '20px 0', borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{borderBottom: '2px solid #333'}}>
            <th style={{padding: '10px', textAlign: 'left'}}>Metric</th>
            <th style={{padding: '10px', textAlign: 'right'}}>CPU</th>
            <th style={{padding: '10px', textAlign: 'right'}}>NPU</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Improvement</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Inference Time</td>
            <td style={{padding: '10px', textAlign: 'right'}}>304.47ms ± 21.67ms</td>
            <td style={{padding: '10px', textAlign: 'right'}}>37.72ms ± 5.81ms</td>
            <td style={{padding: '10px', textAlign: 'right', fontWeight: 'bold', color: '#007700'}}>8.07x faster</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Real-Time FPS</td>
            <td style={{padding: '10px', textAlign: 'right'}}>3.30 FPS</td>
            <td style={{padding: '10px', textAlign: 'right'}}>26.91 FPS</td>
            <td style={{padding: '10px', textAlign: 'right', fontWeight: 'bold', color: '#007700'}}>8.16x faster</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Operating Temp</td>
            <td style={{padding: '10px', textAlign: 'right'}}>49.14°C ± 0.41°C</td>
            <td style={{padding: '10px', textAlign: 'right'}}>48.88°C ± 0.41°C</td>
            <td style={{padding: '10px', textAlign: 'right', fontWeight: 'bold', color: '#007700'}}>0.27°C cooler</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Memory Usage</td>
            <td style={{padding: '10px', textAlign: 'right'}}>~870 MB</td>
            <td style={{padding: '10px', textAlign: 'right'}}>~870 MB</td>
            <td style={{padding: '10px', textAlign: 'right'}}>No overhead</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Detections/Frame</td>
            <td style={{padding: '10px', textAlign: 'right'}}>4.02 ± 1.29</td>
            <td style={{padding: '10px', textAlign: 'right'}}>3.55 ± 1.16</td>
            <td style={{padding: '10px', textAlign: 'right'}}>Accuracy preserved</td>
          </tr>
        </tbody>
      </table>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={performanceComparison} alt="YOLO11 performance comparison" style={{maxWidth: '100%', height: 'auto', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{marginTop: '10px', fontStyle: 'italic', color: '#666', fontSize: '0.9em'}}>
          Performance comparison showing inference time distributions, FPS comparison, temporal stability, and statistical variance
        </p>
      </div>


      <h2>End-to-End System Performance</h2>
      <p>
        The integrated system demonstrates how multiple optimizations compound to create responsive user experiences:
      </p>
      <ol>
        <li><strong>Real-Time Detection:</strong> 27 FPS continuous inference with database logging (37ms per frame)</li>
        <li><strong>Natural Language Processing:</strong> Sub-second LLM response with progressive token streaming</li>
        <li><strong>Query Optimization:</strong> Indexed database searches returning results in {"<"}100ms</li>
        <li><strong>Progressive Rendering:</strong> SSE streaming enables immediate user feedback before completion</li>
      </ol>

      <p>
        End-to-end latency from natural language query to visual results: 2-4 seconds, with perceived latency under 1 second due to streaming. This demonstrates how architectural decisions across the stack (NPU acceleration, async processing, database indexing, streaming updates) combine to achieve production-quality performance.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={rubikpiPhoto} alt="Qualcomm RubikPi  development board" style={{maxWidth: '100%', height: 'auto', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{marginTop: '10px', fontStyle: 'italic', color: '#666', fontSize: '0.9em'}}>
          Qualcomm RubikPi development board with camera attached
        </p>
      </div>

      <h2>Technical Roadmap and Extensions</h2>

      <p>There are a lot of things that I would have loved to add to this project, with more time and resources. Here are some potential extensions:</p>

      <h3>Semantic Search with CLIP Integration</h3>
      <p>
        Current queries are limited to YOLO's 80 object classes. Integrating CLIP embeddings would enable open-vocabulary semantic search ("show me frames with sunset lighting" or "find outdoor scenes"), moving from discrete classification to continuous semantic understanding.
      </p>

      <h3>Event-Driven Alerting System</h3>
      <p>
        Pattern-based monitoring with push notifications ("alert when person detected in zone A" or "notify if car count {">"} 5") would transform passive querying into active surveillance.
      </p>

      <h3>Embedding-Based Similarity Search</h3>
      <p>
        Extracting and indexing visual embeddings enables similarity-based retrieval ("find visually similar frames") and clustering analysis.
      </p>

      <h2>Conclusion</h2>
      <p>
        This project demonstrates the capabilities and usecases of Edge AI technologies. By optimizing a state-of-the-art object detection model with INT8 quantization and deploying it on a Qualcomm NPU, I achieved real-time performance on resource-constrained edge hardware. This acceleration enabled a responsive natural language query system that translates user queries into SQL, executes optimized database searches, and returns visual results with metadata, all while maintaining complete data privacy through local inference.
      </p>
      <p>
        Key metrics: <strong>8x faster inference</strong>, <strong>95%+ SQL generation accuracy</strong>, <strong>27 FPS real-time performance</strong>, <strong>0.27°C cooler operation</strong>, and <strong>sub-second perceived latency</strong> through streaming.
      </p>
      <p>
        Complete code and technical documentation available on <a href="https://github.com/MarcosSaade/smart-vision" target="_blank" rel="noopener noreferrer">GitHub</a>.
      </p>


      <h3>Technology Stack</h3>
      <table style={{margin: '20px 0', borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{borderBottom: '2px solid #333'}}>
            <th style={{padding: '10px', textAlign: 'left'}}>Component</th>
            <th style={{padding: '10px', textAlign: 'left'}}>Technologies</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px', fontWeight: 'bold'}}>Frontend</td>
            <td style={{padding: '10px'}}>React, Server-Sent Events (SSE), Responsive CSS Grid</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px', fontWeight: 'bold'}}>Backend</td>
            <td style={{padding: '10px'}}>FastAPI (Python), SQLite, Async/Await</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px', fontWeight: 'bold'}}>AI/ML</td>
            <td style={{padding: '10px'}}>YOLO11, TensorFlow Lite, INT8 Quantization, Ollama (qwen3:1.7b), Google Gemini</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px', fontWeight: 'bold'}}>Hardware</td>
            <td style={{padding: '10px'}}>Qualcomm QCS6490 SoC, Hexagon Tensor Processor (HTP/NPU), Qualcomm RubikPi Development Board</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px', fontWeight: 'bold'}}>Computer Vision</td>
            <td style={{padding: '10px'}}>OpenCV</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px', fontWeight: 'bold'}}>Infrastructure</td>
            <td style={{padding: '10px'}}>SNPE (Snapdragon Neural Processing Engine), Qualcomm AI Hub</td>
          </tr>
        </tbody>
      </table>

      <h3>System Capabilities</h3>
      <ul>
        <li>Natural language querying with token-by-token streaming responses</li>
        <li>Dual LLM support (local edge AI and cloud AI)</li>
        <li>Real-time video streaming with detection visualization (27 FPS)</li>
        <li>Hardware-accelerated inference on edge devices (8x speedup)</li>
        <li>Intelligent SQL generation with safety validation</li>
        <li>Optimized time-series database queries (5-10x faster with proper indexing)</li>
        <li>80 COCO object classes supported</li>
        <li>Complete privacy option with local-only inference</li>
      </ul>
    </PageTemplate>
  );
}
