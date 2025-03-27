<!-- Template for PROJECT REPORT of CapstoneDesign 2025-2H, initially written by khyoo -->
<!-- 본 파일은 2025년도 컴공 졸업프로젝트의 <1차보고서> 작성을 위한 기본 양식입니다. -->
<!-- 아래에 "*"..."*" 표시는 italic체로 출력하기 위해서 사용한 것입니다. -->
<!-- "내용"에 해당하는 부분을 지우고, 여러분 과제의 내용을 작성해 주세요. -->
<!-- 이미지는 절대경로로 기입 -->

# Team-Info
| (1) 과제명 | __THANOS: Integrating HPO and NAS for Hardware-Aware Transformer Optimization under Memory Size Constraints__
|:---  |---  |
| (2) 팀 번호 / 팀 이름 | 8팀 / eiai |
| (3) 팀 구성원 | __김민서 (2276046)__: 리더, AI, 선행연구 서칭, 연구 환경 세팅; 소프트웨어: 운영체제 설정, 협업 툴 관리 및 지도, 오픈소스 등의 소프트웨어 버전 관리 등; 하드웨어: NPU 장비 관리 및 설치, 실험에 필요한 파서 구현, 연구주제 탐색 및 논문 서칭, 기술 테스트 및 검증, 가설과 문제 제기 및 이에 따른 구현과 실험 진행, 논문 및 보고서 작성 <br> __김수현 (2276067)__: 팀원, AI 관련 연구팀 팀원, 선행 논문 읽고 코드 분석, 선행 연구 코드의 수정 및 보완을 진행, 방법론 관련 논문 서칭, 보고서 작성 <br> __하지연 (2271063)__ : 팀원, AI 연구 팀원, 선행 연구 논문 리딩, 코드 분석, 아이디어를 위한 논문 서칭 및 구현, 보고서 작성|
| (4) 팀 지도교수 | 심재형 교수님 |
| (5) 과제 분류 | 연구 과제 |
| (6) 과제 키워드 | Hardware aware NAS, HPO, Efficient Transformer |
| (7) 과제 내용 요약 |&nbsp; Transformer 모델은 NLP 분야에서 널리 사용되고 있지만, 모델의 크기와 연산량이 증가하면서 높은 컴퓨팅 자원을 요구하게 되었다. 특히 엣지 디바이스와 같은 제한된 환경에서는 모델의 추론에서의 지연과 메모리 사용에서의 제약과 같은 문제가 발생한다. 이러한 문제를 해결하기 위해, 하드웨어 제약을 고려한 경량화 모델 탐색이 필요하다. 본 논문은 NAS(Neural Architecture Search)와 HPO(Hyperparameter Optimization)를 결합한 기법을 통해 이를 해결하고자 한다. 즉, NAS를 통해 구조를 탐색하고, HPO를 통해 성능과 제약 조건 모두를 고려하는 최적의 모델을 찾는다. 특히 NAS의 거대한 서치 스페이스가 초래하는 서치 연산 문제와 메모리 제약조건에 대한 비인지성 문제를 HPO로 보완한다. 이를 통해 주어진 하드웨어에 최적화된 효율적인 트랜스포머 모델을 설계할 수 있다. |

<br>

# Project-Summary
| 항목 | 내용 |
|:---  |---  |
| (1) 문제 정의 | &nbsp;현재 트랜스포머를 비롯한 딥러닝 기반 NLP 인공지능 기술은 우리 일상 속에 깊이 자리 잡고 있다. 그러나 이러한 서비스 대부분은 데이터 센터에서 수많은 반도체를 활용하는 공급자 중심의 클라우드 기반 AI 형태로 제공되고 있다. 그리고 클라우드 기반 AI는 대표적으로 중앙 데이터 센터와의 송수신 과정에서 발생하는 응답지연, 데이터 센터에 사용되는 막대한 유지비용, 보안에서의 취약성과 같은 문제들을 동반한다. <br/> &nbsp;이를 해결하기 위한 대안으로 중앙 클라우드에 대한 의존을 줄이는 엣지 AI가 주목받고 있지만, 현재 딥러닝 모델의 크기와 연산량이 지속적으로 증가하면서 엣지 디바이스의 제한적인 환경에서 모델의 활용에 어려움이 생기고 있다.<br/> &nbsp;이에 본 연구는 대규모 NLP 인공지능을 사용하는 기업이나 엣지 디바이스에서 LLM을 사용하고자 하는 일반 유저를 타겟 고객으로 설정하고, 엣지 디바이스가 가지는 연산능력과 메모리 등 하드웨어 환경적 제약을 고려한 모델 최적화를 실현하고자 한다.|
| (2) 기존연구와의 비교 | &nbsp; __1. 전통적인 모델의 최적화__ <br/>기존의 전통적인 모델 최적화는 전문가의 직관과 경험을 기반으로 하이퍼파라미터나 모델 구조를 수동으로 조정해 성능을 점진적으로 향상시키는 방식이다. 이 방식은 설계자 중심의 직관적 이해와 수정에 용이하다는 장점이 있지만, 트랜스포머와 같은 복잡한 모델에는 적용이 어렵고, 시간과 비용이 많이 드는 단점이 있다. 이를 극복하기 위해 NAS(Neural Architecture Search) 기법이 등장하였고, 다양한 데이터셋과 목적에 맞춰 모델을 자동으로 설계할 수 있게 되었다.<br/><br/>&nbsp; __2. NAS (Neural Architecture Search)__ <br/> NAS는 설계자의 개입 없이 알고리즘적으로 모델 구조를 탐색하며, 복잡한 모델의 설계에서 인간의 직관과 경험에 대한 의존도를 줄여준다. 그러나, 기존의 NAS는 기존 NAS는 주로 FLOPs에 기반한 latency 측정에 의존하고 있으며, 오직 서치 속도나 모델의 정확도에만 집중해왔다. FLOPs에 의존하지 않는 연구들은 주로 동질적인 하드웨어 환경에서만 실험을 거쳐 다양한 하드웨어 환경(예: NPU, FPGA)의 특성을 반영하지 못한다는 한계가 있다. 이는 NAS에서 탐색한 모델의 성능이 실제 하드웨어에서의 성능과 괴리가 생기게 만든다.<br/><br/>&nbsp; __3. HAT (Hardware-aware Transformer)__ <br/> HAT는 하드웨어 환경(CPU, GPU)을 고려하여 트랜스포머 모델을 자동 설계하는 NAS 기반 기법을 제안한 연구로, 기존 NAS에 비해 실제 하드웨어에 맞는 모델을 설계한다는 데 의의가 있다. 그러나 (1) 메모리 제약을 고려하지 않아 latency를 중점적으로 탐색한 결과가 하드웨어 제약 전반을 반영하지 못했고, (2) NPU 환경에서는 적용이 불가능하다는 한계가 있다.<br/><br/>&nbsp; __4.	본 연구의 차별점 및 강점__ <br/>  본 연구는 기존의 한계를 극복하기 위해 다음과 같은 차별점을 가진다:<br/>- NAS를 사용해 하드웨어 친화적인 트랜스포머 모델 구조를 자동으로 탐색한다.<br/>-	FLOPs가 아닌 실제 하드웨어에서의 인퍼런스 실행을 통해 latency를 직접 측정한다.<br/>-	기존 NAS/HAT 연구들이 무시했던 메모리 제약조건을 서치 스페이스에 포함하여 엣지 디바이스 환경에 적합한 실질적 최적화를 실현한다.<br/>-	CPU, GPU뿐 아니라 NPU 환경을 고려한 연구를 진행한다.|
| (3) 제안 내용 | &nbsp;본 연구는 하드웨어 제약 환경에서의 트랜스포머 모델 효율성 문제를 해결하기 위해, NAS(Neural Architecture Search)와 HPO(Hyperparameter Optimization)를 결합한 자동화된 최적화 기법을 제안한다. <br/> &nbsp; 먼저, FLOPs 기반의 이론적 성능 추정이 실제 지연시간(latency)을 정확히 반영하지 못한다는 점을 고려하여, 실제 하드웨어에서 직접 인퍼런스를 수행하는 NAS 구조를 설계하였다. 이를 통해 각 하드웨어 환경에 맞춘 정확한 성능 측정 및 평가가 가능하도록 하였다. <br/>&nbsp; 그러나, 특수 프로세서 환경에서는 하드웨어 메모리 제약으로 인해 NAS 탐색 과정에서 모델 인퍼런스 실행이 불가능해지는 문제가 발생했다. 이에 대한 해결책으로 HPO 기법을 도입하여, NAS의 방대한 서치 공간에서 발생하는 탐색 효율 저하 문제와 메모리 비인지성 문제를 동시에 해결하고자 한다. <br/>&nbsp; 또한, 기존 연구들이 주로 CPU나 GPU 환경에 한정되어 있던 것과 달리, 본 연구에서는 NPU와 같은 AI 전용 프로세서 환경까지 고려할 수 있도록 모델 변환(model converting) 단계를 추가하였다.<br/>&nbsp; 이러한 일련의 접근을 통해, 본 연구는 성능 저하 없이 모델 크기를 줄이고, 추론 속도를 향상시키는 하드웨어 친화적 트랜스포머 모델 탐색을 실현하고자 한다. |
| (4) 기대효과 및 의의 | &nbsp;본 연구를 통해 개발된 NPU 구조에 최적화되고 경량화된 트랜스포머 모델은, 연산 자원이 제한된 엣지 디바이스 환경에서도 딥러닝 모델이 효율적으로 동작할 수 있도록 지원한다. 이를 통해 기존 클라우드 기반 AI가 안고 있던 과도한 에너지 소비, 보안 취약성, 높은 운영 비용 등의 문제를 완화하고, 보다 경제적이고 안전한 AI 서비스 운영이 가능해질 것으로 기대된다.<br/>&nbsp; 또한, 본 연구에서 제안한 NAS+HPO 기반 최적화 기법은 특정 모델에 국한되지 않고, 다른 딥러닝 모델의 최적화에도 확장 적용될 수 있는 범용적인 프레임워크로서의 가능성을 가진다. 더 나아가, NPU뿐만 아니라 다양한 하드웨어 환경에서도 적용 가능성이 열려 있어, 엣지 AI의 산업적 활용 범위를 넓히는 데 기여할 수 있을 것이다. |
| (5) 주요 기능 리스트 |&nbsp; __1. Hardware aware NAS 프레임워크__ <br/> 모델 구조를 자동으로 탐색하는 NAS 기법을 기반으로 하되, 하드웨어 환경의 제약(지연시간, 메모리 크기 등)을 고려할 수 있도록 프레임워크를 설계한다. <br/> 구성 요소는 다음과 같다:<br/> - 하드웨어 환경적 제약을 반영한 탐색 공간(Search Space) 정의<br/>-	탐색 공간에서 최적의 아키텍처를 탐색하는 (Search Strategy) 수립 <br/>-	각 아키텍처의 성능 평가(Performance Estimator)<br/><br/>&nbsp; __2. 실제 하드웨어에서의 인퍼런스 기반 성능 평가__ <br/>기존 NAS가 FLOPs로 latency를 추정하는 것과 달리, 실제 하드웨어(NPU 등)에서 직접 인퍼런스를 실행해 latency를 측정한다. 이를 기반으로 latency dataset을 생성하고, latency 제약조건을 만족하는 트랜스포머 모델의 구조를 탐색한다.<br/><br/>&nbsp; __3. 메모리 제약 조건을 반영한 탐색 공간 구현__ <br/> HAT의 한계였던 메모리 제약 미반영 문제를 해결하기 위해, 탐색 공간 내에서 메모리 제약 조건을 직접 반영한다. 이를 위해 트랜스포머의 주요 구성요소인 QKV 차원(qkv dim)을 조정 가능한 변수로 설정한다. QKV dim은 self-attention 연산에서 메모리 사용량에 직접적인 영향을 미치기 때문에, 이를 축소하면 모델 전체의 메모리 사용량을 효과적으로 줄일 수 있다. 탐색 공간에서 메모리 제약 조건의 설정은 사전 정의된 하드웨어 메모리 한도 이내의 아키텍처만 탐색 대상으로 포함하게 한다. 이로써 탐색 효율성과 실행 가능성을 모두 확보할 수 있다.<br/><br/>&nbsp; __4. 다양한 하드웨어 환경(NPU 포함) 호환을 위한 모델 변환 기능__ <br/> 훈련을 진행할 Supertransformer가 실제 다양한 하드웨어 환경에서 실행 가능하도록 모델 변환 과정을 수행한다. 구체적으로 PyTorch → ONNX → RKNN 변환 흐름을 따르며, 이는 특히 NPU에서의 실행을 지원하기 위함이다. <br/> 먼저, Pytorch 로 구현된 supertransformer 모델을 ONNX 형식으로 변환함으로써, 프레임워크 간의 호환성을 확보한다. ONNX는 다양한 하드웨어 백엔드와도 연결될 수 있어, 향후 NPU 이외의 다양한 엣지나 서버 환경에서도 실행가능성을 확보할 수 있다. 이후 ONNX 모델을 Rknn toolkit을 통해 변환함으로써 Rockchip 기반 NPU에서 직접 실행 가능한 모델로 최종 변환한다.|

<br>
 
# Project-Design & Implementation
| 항목 | 내용 |
|:---  |---  |
| (1) 요구사항 정의 | *프로젝트를 완성하기 위해 필요한 요구사항을 설명하기에 가장 적합한 방법을 선택하여 기술* <br> 예) <br> - 기능별 상세 요구사항(또는 유스케이스) <br> - 설계 모델(클래스 다이어그램, 클래스 및 모듈 명세서) <br> - UI 분석/설계 모델 <br> - E-R 다이어그램/DB 설계 모델(테이블 구조) |
| (2) 전체 시스템 구성 | 다음은 본 연구에서 사용하고자 하는 NAS 알고리즘을 도식화한 것이다. <br/> <br> &nbsp; __1.	Search space__ <br> Search space는 알고리즘이 탐색을 수행하는 공간을 의미한다. NAS는 이 공간 내에서 최적의 모델 아키텍처를 탐색한다. 탐색 공간은 탐색할 수 있는 모든 모델 구조들의 집합으로, 어떤 연산을 쓸지, 연산들이 어떻게 연결될지 등이 정의되어 있다. 본 연구에서는 다음과 같은 구성 요소를 포함한 탐색 공간을 정의한다: <br> -	Arbitrary Encoder-Decoder Attention <br> 모든 encoder layer 를 decoder가 자유롭게 참조할 수 있도록 한다. <br> -	Heterogeneous Transformer Layers <br> 각 layer 별로 hidden size, head 수, FFN 크기 등이 다르게 구성될 수 있어, 다양한 하드웨어 특성에 맞춘 유연한 아키텍처 설계가 가능하다. <br> -	Hyperparameter (QKV Dimension) <br> 각 하드웨어에 최적화된 qkv dim을 조정하도록 구현하면서 메모리 사이즈를 고려한 트랜스포머의 탐색이 가능하도록 한다.<br><br> &nbsp; __2.	Search Strategy__ <br> Search Strategy는 탐색 방법으로 Search space 내의 어떤 아키텍처를 선택할지 결정하는 전략이다. 탐색 방법으로는 강화학습, 진화 알고리즘, 그래디언트 방식 등 여러 알고리즘이 쓰인다. 본 연구에서는 Supertransformer라는 모든 Subtransformer들을 포함하는 큰 네트워크를 한 번 훈련시키고, 진화 알고리즘(Evolutionary Search)를 통해 하드웨어 latency가 가장 낮고 성능이 좋은 Subtransformer를 탐색한다. 또한, HPO(Hyperparameter Optimization)을 통해 하드웨어에 최적화된 qkv dim 구성을 선택할 수 있도록 한다.<br><br> &nbsp; __3.	Performance Estimation Strategy__ <br> Performance Estimation Strategy는 선택된 아키텍처의 성능을 평가하거나 추정하는 방법이다. 보통 NAS는 search strategy가 하나 혹은 여러 개의 후보 아키텍처를 추출하고, 이들의 성능을 예측한다. 본 연구에서는 다음과 같은 성능 추정 방법을 채택한다: <br> -	weight sharing <br> subtransformer의 개별 학습 없이 이미 학습된 supertransformer의 weight를 사용하여 빠른 성능 평가가 가능한다. <br> -	하드웨어 기반 Latency 측정<br> FLOPs 대신 실제 하드웨어에서의 지연시간을 측정하여 플랫폼 맞춤 성능 평가가 가능하다. <br> *** ### ✅ 실험 환경 구성

- **보드**  
  ROCK Pi 5B (Rockchip RK3588 ARM SoC 기반 NPU 보드)

- **운영체제 (OS)**  
  - Ubuntu 22.04 (aarch64)  
  - Debian (Radxa Rock 5B 공식 이미지)

- **모델 아키텍처**  
  Transformer 기반 구조

- **사용 데이터셋**  
  - WMT’14 En-De  
  - WMT’14 En-Fr  
  - WMT’19 En-De  
  - IWSLT’14 De-En

- **사용 Python 라이브러리 목록**
  ```
  rknn-toolkit2==2.3.0
  rknn-toolkit-lite2==1.5.0
  charset-normalizer==3.4.0
  datasets==3.1.0
  fast-histogram==0.13
  flatbuffers==24.3.25
  huggingface-hub==0.26.5
  mpmath==1.3.0
  multidict==6.1.0
  multiprocess==0.70.16
  numpy==1.24.4
  onnx==1.14.1
  onnxoptimizer==0.2.7
  onnxruntime==1.16.0
  opencv-python==4.10.0.84
  packaging==24.2
  pandas==2.0.3
  safetensors==0.4.5
  scipy==1.10.1
  sympy==1.13.3
  tokenizers==0.20.3
  torch==2.4.1
  torchdata==0.7.1
  torchtext==0.15.2
  torchvision==0.19.1
  tqdm==4.67.1
  transformers==4.46.3
  ```
 |
| (3) 주요엔진 및 기능 설계 | *프로젝트의 주요 기능 혹은 모듈의 설계내용에 대하여 기술한다 <br> SW 구조 그림에 있는 각 Module의 상세 구현내용을 자세히 기술한다.* |
| (4) 주요 기능의 구현 | *<주요기능리스트>에 정의된 기능 중 최소 2개 이상에 대한 상세 구현내용을 기술한다.* |
| (5) 기타 | 참고문헌:   Thomas Elsken, Jan Hendrik Metzen, Frank Hutter, "Neural Architecture Search: A Survey," arXiv preprint arXiv:1808.05377, 2019.    Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, Song Han, "HAT: Hardware-Aware Transformers for Efficient Natural Language Processing," arXiv preprint arXiv:2005.14187, 2020. |

<br>
