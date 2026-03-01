# Java (Spring Boot) API 호출 연동 가이드

본 문서는 사내 레거시 시스템이나 Spring Boot 기반의 백엔드 서버에서 현재 구축된 파이썬 AI 서버(STT+요약)로 `.wav` 파일을 전송하고, **텍스트 전문과 3줄 요약 결과를 돌려받는 방법**에 대해 설명합니다.

---

## 1. 연동 고려사항 및 주의점 (필독)

1. **타임아웃(Timeout) 설정 필수:**
   STT 변환과 LLM 요약 작업은 파일 길이에 따라 **최소 10초에서 수 분 이상** 소요될 수 있습니다. Spring Boot의 기본 다운로드/읽기 타임아웃은 보통 짧게 잡혀 있으므로, `RestTemplate` 객체 생성 시 **ReadTimeout을 최소 3분(180초) 이상으로 넉넉하게 설정**해야 중간에 연결이 끊기는 현상을 방지할 수 있습니다.
2. **응답 포맷 (JSON):**
   정상적으로 처리된 경우(`HTTP 200 OK` 또는 `207 Partial Content`), 응답 JSON 내부에 `transcript`(전문)와 `summary`(3줄 요약) 필드가 담겨서 내려옵니다.

---

## 2. Spring Boot 연동 코드 예제

Spring Boot 웹 환경에서 기본적으로 제공하는 `RestTemplate`과 JSON 파싱용 `Jackson ObjectMapper`를 활용한 예시입니다.

```java
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;

@Service
public class SttClientService {

    public void processAudioFile() {
        // 1. 타임아웃이 설정된 RestTemplate 생성 (핵심)
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(10000); // 연결 타임아웃: 10초
        factory.setReadTimeout(300000);   // 읽기 타임아웃: 300초 (5분) - 모델 인퍼런스 시간 대기
        RestTemplate restTemplate = new RestTemplate(factory);
        
        // 파이썬 AI 서버 주소 (실제 서버 IP 포트로 변경)
        String aiServerUrl = "http://192.168.1.100:9123/api/v1/process";

        // 2. HTTP 헤더 설정 (Multipart-Form)
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // 3. 바디(Body)에 전송할 파일 및 파라미터 담기
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        
        // 실제 존재하는 WAV 파일의 경로 설정
        File audioFile = new File("/경로/내파일.wav"); 
        body.add("file", new FileSystemResource(audioFile));
        body.add("language", "ko");
        
        // (선택) 청크 옵션 활성화 시 같이 담습니다.
        // body.add("chunk_enabled", true);

        // 4. 요청 엔티티 조립
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            // 5. API 서버로 POST 요청 전송 (응답은 String 방식의 JSON 리턴)
            System.out.println("AI 서버에 분석을 요청합니다. 잠시만 기다려주세요...");
            ResponseEntity<String> response = restTemplate.postForEntity(aiServerUrl, requestEntity, String.class);
            
            // 6. 응답 JSON 파싱 (Jackson ObjectMapper 활용)
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response.getBody());

            // 7. 반환된 JSON 데이터 추출 및 활용
            if ("success".equals(root.get("status").asText())) {
                String fullTranscript = root.get("transcript").asText();
                String llmSummary = root.get("summary").asText();
                double totalTimeSec = root.get("processing_time").get("total_sec").asDouble();

                System.out.println("==========================================");
                System.out.println("[STT 텍스트 전문]");
                System.out.println("==========================================");
                System.out.println(fullTranscript);
                
                System.out.println("\n==========================================");
                System.out.println("[LLM 3줄 요약]");
                System.out.println("==========================================");
                System.out.println(llmSummary);
                
                System.out.println("\n(총 소요 시간: " + totalTimeSec + "초)");
            } else {
                System.out.println("API 처리 실패 또는 에러 발생: " + root.get("error").asText());
            }

        } catch (Exception e) {
            System.err.println("API 호출 중 예외 발생: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
```

---

## 3. 반환(Response) 받는 JSON 스펙

위 코드가 성공적으로 실행되었을 때, `response.getBody()`로 날아오는 실제 JSON 데이터의 모습은 아래와 같습니다.

```json
{
  "status": "success",
  "transcript": "상담사: 고객님 안녕하세요. 무엇을 도와드릴까요?\n고객: 비밀번호를 까먹었어요.",
  "segments": [
    {
      "start": 0.0,
      "end": 3.4,
      "text": "상담사: 고객님 안녕하세요. 무엇을 도와드릴까요?"
    },
    {
      "start": 3.5,
      "end": 6.8,
      "text": "고객: 비밀번호를 까먹었어요."
    }
  ],
  "summary": "[주제] 비밀번호 분실 문의\n[상담원 대응] 비밀번호 초기화 안내",
  "model_info": {
    "stt": "mlx-whisper/small",
    "llm": "qwen2.5:7b"
  },
  "processing_time": {
    "stt_sec": 4.5,
    "llm_sec": 6.2,
    "total_sec": 10.7
  },
  "error": null
}
```
