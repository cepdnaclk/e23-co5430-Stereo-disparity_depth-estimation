"""Startup regressions: run with python -m unittest discover -s tests."""

import os
import unittest
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import cv2
import gradio
import yaml
from fastapi.testclient import TestClient
from gradio_client.utils import json_schema_to_python_type

import app


class AppStartupTests(unittest.TestCase):
    def test_space_and_local_gradio_versions_match(self):
        root = Path(app.current_dir)
        metadata = yaml.safe_load((root / "README.md").read_text(encoding="utf-8").split("---", 2)[1])
        self.assertEqual(metadata["sdk_version"], gradio.__version__)
        self.assertIn(
            f"gradio=={gradio.__version__}",
            (root / "requirements.txt").read_text(encoding="utf-8").splitlines(),
        )

    def test_boolean_schema(self):
        # Pydantic emits this for free-form file metadata.
        schema = {"type": "object", "additionalProperties": True}
        self.assertIsInstance(json_schema_to_python_type(schema), str)

    def test_homepage_and_api_schema(self):
        # Exercise real Image, ImageSlider and File schemas, including callbacks.
        info = app.demo.get_api_info()
        self.assertIn("/process_single_stereo", info["named_endpoints"])
        with TestClient(app.demo.app) as client:
            for path in ("/", "/config", "/gradio_api/info"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 200)

    def test_sample_stereo_slider_output(self):
        examples = Path(app.current_dir) / "examples"
        left = cv2.cvtColor(cv2.imread(str(examples / "sample_left.png")), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(str(examples / "sample_right.png")), cv2.COLOR_BGR2RGB)
        for model, block_size in (("StereoSGBM", 3), ("StereoBM", 25)):
            with self.subTest(model=model):
                images, stats, disparity = app.process_single_stereo(
                    left, right, model, block_size, 160, 10, "plasma"
                )
                self.assertEqual(disparity.shape, left.shape[:2])
                self.assertTrue((disparity > 0).any())
                self.assertIn(model, stats)
                self.assertIsNotNone(app.slider_out.postprocess(images))


if __name__ == "__main__":
    unittest.main()
