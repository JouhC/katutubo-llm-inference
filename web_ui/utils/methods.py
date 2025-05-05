import time
import requests

class KatutuboLLMAPI:
    def __init__(self, base_url, max_retries=20, retry_delay=5):
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        response.raise_for_status()
    
    def _wait_for_service(self):
        """Wait until the service is up before making any requests."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{self.base_url}/healthz", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                print(f"Waiting for API... (Attempt {attempt + 1}/{self.max_retries})")
                time.sleep(self.retry_delay)
        print("API did not start in time.")
        return False

    
    def _safe_request(self, method, endpoint, data=None):
        """Ensure the service is running before making the request."""
        if not self._wait_for_service():
            raise Exception("API service is not available.")

        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                print(data)
                response = requests.post(url, json=data)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    # Root endpoint
    def get_root(self):
        return self._safe_request("GET", "/")
    
    def infer(self, prompt, history):
        return self._safe_request("POST", "/infer", {"prompt": prompt, "history": history})
