# Social Leads Agent

An AI-powered chatbot agent for Instagram that manages leads, schedules meetings, and provides information about your company's portfolio using advanced language models and integrations.

## Features

- **Instagram Integration**: Handles incoming messages via Instagram Messenger API
- **Customer Management**: Stores and retrieves customer information using Airtable
- **Meeting Scheduling**: Integrates with Google Calendar to schedule and manage meetings
- **RAG-Powered Q&A**: Answers questions about your portfolio/projects using Retrieval-Augmented Generation
- **Session Management**: Maintains conversation history per user
- **Lead Qualification**: Automatically extracts customer details from conversations

## Architecture

The project consists of:
- `main.py`: Flask web server handling Instagram webhooks
- `social_agent_v11.py`: Main AI agent with full capabilities (RAG, Calendar, Customer context)
- `social_agent_v10.py`: Simplified calendar-only agent
- `airtable_tool.py`: Airtable integration for customer data
- `__init__.py`: Package initialization

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Instagram Business Account
- Google Cloud Project with Calendar API enabled
- Airtable account and base
- Groq API key

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/arslandevs/social_leads_agent.git
cd social_leads_agent
```

Install dependencies:
```bash
pip install -U langchain langchain-community langchain-core langchain-groq \
  langchain-google-genai chromadb langgraph pypdf python-dotenv \
  langchain-google-community pyairtable flask requests
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Airtable Configuration
AIRTABLE_TOKEN=your_airtable_token_here
AIRTABLE_BASE_ID=your_base_id_here
AIRTABLE_TABLE=customers

# Optional: Debug Airtable
AIRTABLE_DEBUG=0
AIRTABLE_VIEW=your_view_name
```

### 4. Google Calendar Setup

1. Create a Google Cloud Project and enable the Calendar API
2. Create OAuth 2.0 credentials (download `credentials.json`)
3. Run the authentication flow to generate `token.json`

Place both `credentials.json` and `token.json` in the project root.

### 5. Airtable Setup

1. Create an Airtable base with a "customers" table
2. Add fields: Name, Email, Username, Description
3. Generate a Personal Access Token
4. Update the `.env` file with your token and base ID

### 6. Portfolio Documents (Optional)

For RAG functionality, place a PDF file named `portfolio_projects.pdf` in the root directory containing your company's portfolio information.

### 7. Instagram Setup

1. Set up an Instagram Business Account
2. Create a Facebook App with Messenger API
3. Configure webhooks pointing to your deployed `/webhook` endpoint
4. Update the tokens in `main.py`:
   - `codestreaks_user_token`
   - `IG_PAGE_ACCESS_TOKEN`
   - `VERIFY_TOKEN`
   - `PAGE_ID`
   - `IG_SCOPED_USER_ID`

## Running the Application

Start the Flask server:
```bash
python main.py
```

The server will run on `http://0.0.0.0:5000`

For production, deploy to a server with a public URL and configure Instagram webhooks accordingly.

## Usage

Once set up, users can message your Instagram account and the agent will:

- Respond to inquiries about your services/portfolio
- Schedule meetings (with confirmation)
- Store customer information
- Maintain conversation context

## Agent Versions

- **v10**: Basic calendar scheduling agent
- **v11**: Full-featured agent with RAG, customer management, and calendar integration

The main application uses v11 by default.

## Security Notes

- Never commit API keys or tokens to version control
- Use environment variables for all sensitive data
- The `.gitignore` file excludes sensitive files like `.env`, `credentials.json`, `token.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license here]