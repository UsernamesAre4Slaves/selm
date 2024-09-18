import unittest
import torch
from src.tasks.text_classification import TextClassificationDataset, train, evaluate, predict
from src.tasks.summarization import SummarizationDataset, summarize_text
from src.tasks.question_answering import QuestionAnsweringDataset, answer_question

class TestTextClassification(unittest.TestCase):
    def setUp(self):
        # Set up a simple dataset and model for text classification testing
        self.texts = ["This is a sample text.", "Another sample text."]
        self.labels = [0, 1]
        self.dataset = TextClassificationDataset(self.texts, self.labels)
        self.dummy_input = torch.randn(2, 10, 256)  # Dummy input for testing
        self.dummy_labels = torch.tensor([0, 1])     # Dummy labels for testing

    def test_dataset(self):
        """Test TextClassificationDataset."""
        self.assertEqual(len(self.dataset), len(self.texts))
        item = self.dataset[0]
        self.assertEqual(item['text'], self.texts[0])
        self.assertEqual(item['label'], self.labels[0])

    def test_train(self):
        """Test the train function for text classification."""
        model = torch.nn.Linear(256, 2)  # Dummy model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dummy_dataloader = [(self.dummy_input, self.dummy_labels)]

        avg_loss, accuracy = train(model, dummy_dataloader, criterion, optimizer, 'cpu')
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_evaluate(self):
        """Test the evaluate function for text classification."""
        model = torch.nn.Linear(256, 2)  # Dummy model
        criterion = torch.nn.CrossEntropyLoss()
        dummy_dataloader = [(self.dummy_input, self.dummy_labels)]

        avg_loss, accuracy = evaluate(model, dummy_dataloader, criterion, 'cpu')
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_predict(self):
        """Test the predict function for text classification."""
        model = torch.nn.Linear(256, 2)  # Dummy model
        prediction = predict(model, self.dummy_input[0].tolist(), 'cpu')
        self.assertIn(prediction, [0, 1])

class TestSummarization(unittest.TestCase):
    def setUp(self):
        self.text = "This is a long document that needs to be summarized."
        self.summarized_text = summarize_text(self.text)
    
    def test_summarize_text(self):
        """Test the summarize_text function."""
        self.assertIsInstance(self.summarized_text, str)
        self.assertGreater(len(self.summarized_text), 0)
        self.assertNotEqual(self.summarized_text, self.text)

class TestQuestionAnswering(unittest.TestCase):
    def setUp(self):
        self.context = "The capital of France is Paris."
        self.question = "What is the capital of France?"
        self.answer = answer_question(self.context, self.question)

    def test_answer_question(self):
        """Test the answer_question function."""
        self.assertIsInstance(self.answer, str)
        self.assertEqual(self.answer, "Paris")

if __name__ == "__main__":
    unittest.main()
