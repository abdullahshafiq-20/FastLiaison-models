import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EmailClassifier:
    """
    BERT-based Email Classifier for Job-related emails
    """
    def __init__(self, model_path='best_bert_email_classifier.pth', max_length=128):
        """
        Initialize the email classifier
        
        Args:
            model_path (str): Path to the saved model weights
            max_length (int): Maximum sequence length for tokenization
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        print(f"Loading model on device: {self.device}")
        
        # Load tokenizer
        print("Loading BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load model
        print("Loading BERT model...")
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Load trained weights
        print(f"Loading model weights from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!\n")
    
    def predict(self, text):
        """
        Predict if an email is job-related or not
        
        Args:
            text (str): Email content to classify
            
        Returns:
            dict: Dictionary containing prediction, confidence, and probabilities
        """
        try:
            # Tokenize input
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = float(probabilities[0][prediction].item() * 100)
            
            # Get probabilities for both classes - ensure they're Python floats
            not_job_prob = float(probabilities[0][0].item() * 100)
            job_prob = float(probabilities[0][1].item() * 100)
            
            # Clamp confidence values to ensure they're within [0, 100] due to floating point precision
            confidence = max(0.0, min(100.0, confidence))
            not_job_prob = max(0.0, min(100.0, not_job_prob))
            job_prob = max(0.0, min(100.0, job_prob))
            
            # Ensure prediction is a Python int
            prediction_int = int(prediction)
            
            result = {
                'prediction': 'job' if prediction_int == 1 else 'not_job',
                'confidence': confidence,
                'probabilities': {
                    'not_job': not_job_prob,
                    'job': job_prob
                },
                'label': prediction_int
            }
            
            logger.debug(f"Prediction result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
    
    def predict_batch(self, texts):
        """
        Predict multiple emails at once
        
        Args:
            texts (list): List of email contents to classify
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def print_prediction(self, text, result=None):
        """
        Print prediction in a formatted way
        
        Args:
            text (str): Email content
            result (dict): Prediction result (if None, will compute it)
        """
        if result is None:
            result = self.predict(text)
        
        print("=" * 70)
        print("EMAIL CLASSIFICATION RESULT")
        print("=" * 70)
        print(f"\nEmail Content (first 200 chars):")
        print(f"{text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"\n{'-' * 70}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"\nProbabilities:")
        print(f"  - Not Job: {result['probabilities']['not_job']:.2f}%")
        print(f"  - Job:     {result['probabilities']['job']:.2f}%")
        print("=" * 70)


def main():
    """
    Main function to demonstrate usage
    """
    # Initialize classifier
    classifier = EmailClassifier(model_path='best_bert_email_classifier.pth')
    
    # Example emails to test
    test_emails = [
        """
        Subject: Senior Software Engineer Position - Tech Corp
        
        Dear Candidate,
        
        We are excited to announce an opening for a Senior Software Engineer position 
        at Tech Corp. We are looking for someone with 5+ years of experience in Python 
        and machine learning. The role offers competitive salary and benefits.
        
        Please submit your resume to careers@techcorp.com
        
        Best regards,
        HR Team
        """,
        
        """
        Subject: Weekend Plans
        
        Hey!
        
        Are you free this weekend? I was thinking we could grab dinner and catch up. 
        Let me know what works for you!
        
        Cheers,
        Sarah
        """,
        
        """
        Subject: Job Opportunity - Data Scientist Role
        
        Hello,
        
        I came across your profile and think you'd be a great fit for our Data Scientist 
        position. We're offering a remote position with excellent benefits. 
        
        Interested? Reply to this email with your CV.
        
        Thanks,
        Recruiter
        """,
        
        """
        Subject: Meeting Reminder
        
        Hi team,
        
        Just a reminder about our meeting tomorrow at 2 PM. Please review the documents 
        I sent earlier and come prepared with your feedback.
        
        See you tomorrow!
        """,
        
        """
        Subject: Freelance Developer Needed
        
        We are looking for a freelance web developer to help build our e-commerce platform. 
        Must have experience with React and Node.js. Project duration: 3 months.
        Rate: $50/hour. Apply now!
        """
    ]
    
    print("\n" + "=" * 70)
    print("TESTING EMAIL CLASSIFIER WITH SAMPLE EMAILS")
    print("=" * 70 + "\n")
    
    # Predict each email
    for i, email in enumerate(test_emails, 1):
        print(f"\n{'*' * 70}")
        print(f"TEST EMAIL #{i}")
        print(f"{'*' * 70}\n")
        
        result = classifier.predict(email)
        classifier.print_prediction(email, result)
        
        print()  # Extra line for spacing
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nEnter your own email to classify (or 'quit' to exit):\n")
    
    while True:
        user_input = input("\nEnter email content (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            print("\nExiting. Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text!")
            continue
        
        result = classifier.predict(user_input)
        classifier.print_prediction(user_input, result)


if __name__ == "__main__":
    main()
