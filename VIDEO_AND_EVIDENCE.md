# Video and evidence guide

No video or runtime screenshots are fabricated or bundled. Record a 3–5 minute walkthrough after executing the project.

Suggested sequence:
1. Show repository structure and dataset download.
2. Run EDA and open the generated plots.
3. Open MLflow and show both experiment runs and metrics.
4. Run `pytest -q` and show the result.
5. Build and run Docker; demonstrate `/health`, `/docs`, and a sample `/predict`.
6. Show Prometheus target status and query a request metric.
7. Show Minikube pod/service and port-forwarded health endpoint.
8. Summarize limitations and mention this is not for medical decisions.

Save real screenshots in `screenshots/` and link the video in the report and README.
