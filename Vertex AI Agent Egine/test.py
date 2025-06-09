import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

# Add the app directory to the path to import main
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import fetch_pdf, classify_document, DocumentState

class TestWorkflowNodes(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test artifacts."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup) # Ensures cleanup even if tests fail

    @patch('main.storage_client')
    def test_fetch_pdf_success(self, mock_storage_client):
        """Tests if fetch_pdf successfully 'downloads' a file."""
        # Arrange: Mock the GCS bucket and blob interactions
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # The state to be passed to the function
        doc_id = "test/document.pdf"
        state = DocumentState(document_id=doc_id)

        # Act: Call the function
        result_state = fetch_pdf(state, self.test_dir.name)

        # Assert: Check if the GCS client was called correctly
        mock_storage_client.bucket.assert_called_with("domain-evonance-storage")
        mock_bucket.blob.assert_called_with(f"source_documents/{doc_id}")

        # Assert: Check that download_to_filename was called with the correct path
        expected_local_path = os.path.join(self.test_dir.name, doc_id)
        mock_blob.download_to_filename.assert_called_with(expected_local_path)

        # Assert: Check if the state was updated correctly
        self.assertEqual(result_state['pdf_path'], expected_local_path)

    @patch('main.genai')
    def test_classify_document_success(self, mock_genai):
        """Tests if classify_document correctly processes a mock API response."""
        # Arrange: Mock the GenAI model and its response
        mock_model_response = MagicMock()
        mock_model_response.text = "pay"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_model_response

        # Mock the file upload/delete process
        mock_file = MagicMock()
        mock_file.name = "files/test-pdf"
        mock_genai.upload_file.return_value = mock_file

        # The input state with a dummy pdf_path
        state = DocumentState(
            pdf_path="/fake/path/to/payslip.pdf",
            document_id="payslip.pdf"
        )

        # Act: Call the function
        result_state = classify_document(state)

        # Assert: Check if the state was updated correctly
        self.assertEqual(result_state['doc_type'], 'pay')
        # Assert that the temporary uploaded file was deleted
        mock_genai.delete_file.assert_called_with("files/test-pdf")

    def test_classify_document_no_pdf_path(self):
        """Tests the fallback behavior when no PDF path is provided."""
        # Arrange: Input state with no pdf_path
        state = DocumentState(pdf_path="", document_id="failed_download.pdf")

        # Act: Call the function
        result_state = classify_document(state)

        # Assert: Check that it defaults to 'bank' as per the logic
        self.assertEqual(result_state['doc_type'], 'bank')

if __name__ == '__main__':
    unittest.main()