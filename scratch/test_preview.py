import httpx
import time

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=120)

def run_preview(text: str, lang: str = "en") -> float:
    t0 = time.perf_counter()
    resp = client.post(
        "/v1/tts/preview",
        json={"voice_id": "client_ref", "text": text, "language": lang, "speed": 1.0},
    )
    post_time = time.perf_counter() - t0
    data = resp.json()
    job_id = data.get("job_id")
    print(f"POST status={resp.status_code}, job_id={job_id}, post_time={post_time:.3f}s")
    
    start_poll = time.perf_counter()
    while True:
        status_resp = client.get(f"/v1/jobs/{job_id}")
        status_data = status_resp.json()
        st = status_data.get("status")
        if st in ("completed", "failed"):
            total_time = time.perf_counter() - t0
            poll_duration = time.perf_counter() - start_poll
            stage = status_data.get("stage", "Unknown")
            print(f"Final status={st}, total_time={total_time:.3f}s (poll={poll_duration:.3f}s), stage={stage}")
            if st == "failed":
                print("Error:", status_data.get("error_message"))
            else:
                out_meta = status_data.get("output_metadata", {})
                print(f"Audio duration: {out_meta.get('duration_seconds')}s")
            return total_time
        time.sleep(0.5)

if __name__ == "__main__":
    print("=== TEST 1 (Cold): English Preview ===")
    t1 = run_preview("Hello, welcome to AudioPipeline V1 studio.", "en")

    print("\n=== TEST 2 (Warm): English Preview ===")
    t2 = run_preview("We are testing ultra low latency preview generation.", "en")

    print(f"\nRESULTS:\nCold Preview Time: {t1:.2f}s\nWarm Preview Time: {t2:.2f}s")
