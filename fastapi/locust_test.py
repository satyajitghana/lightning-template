from locust import HttpUser, task, between


class StressTest(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def test_text_endpoint(self):
        url = "/text-to-image?text=a dog playing on a beach"
        res = self.client.get(
            url=url,
            headers={},
            data={}
        )
