# AI Emblematic

A Flask web application for intelligent document processing with AI-powered analysis and FHIR conversion capabilities.

## Features

- **Document Upload**: PDF file upload and text extraction
- **AI Analysis**: OpenAI-powered document summarization and classification
- **FHIR Conversion**: Convert medical documents to FHIR format
- **Smart Caching**: Avoid re-processing the same documents
- **Bootstrap 5**: Modern, responsive UI components
- **Template Inheritance**: Consistent layout using Jinja2 templates
- **API Endpoints**: RESTful API for data interaction
- **Error Handling**: Comprehensive error handling and validation

## Project Structure

```
emblematic/
├── app.py                 # Main Flask application
├── models.py              # Pydantic models for AI analysis
├── keys.py                # API keys (not in version control)
├── keys_example.py        # Example API keys file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore file
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── about.html        # About page
│   └── text_display.html # Document display page
├── static/               # Static files
    ├── online-A-long-journey.png  # Background image
├── uploads/              # Uploaded files (not in version control)
└── text_cache/           # Cached text and analysis (not in version control)
    │   └── main.js       # JavaScript functionality
    └── images/           # Image assets
```

## Installation

1. **Clone or download** this project to your local machine

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   - Copy `keys_example.py` to `keys.py`
   - Add your OpenAI API key to `keys.py`
   - **Important**: Never commit your actual API keys to version control!

## API Key Configuration

The application requires an OpenAI API key for AI analysis and FHIR conversion features:

1. **Create `keys.py`**: Copy `keys_example.py` to `keys.py`
2. **Add your API key**: Replace `"your_openai_api_key_here"` with your actual OpenAI API key
3. **Security**: The `keys.py` file is automatically ignored by git to prevent accidental commits

Example `keys.py`:
```python
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

**Note**: Never commit your actual API keys to version control!

## Running the Application

1. **Start the Flask development server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## API Endpoints

### GET /api/hello
Returns a simple greeting message.

**Response:**
```json
{
  "message": "Hello from Flask!",
  "status": "success"
}
```

### POST /api/echo
Echoes back the JSON data sent in the request.

**Request Body:**
```json
{
  "message": "Hello",
  "data": "example"
}
```

**Response:**
```json
{
  "received": {
    "message": "Hello",
    "data": "example"
  },
  "message": "Data received successfully"
}
```

## Pages

- **Home** (`/`): Welcome page with feature overview and API testing
- **About** (`/about`): Information about the application and technologies used
- **404**: Custom page for not found errors
- **500**: Custom page for server errors

## Development

### Adding New Routes

Add new routes in `app.py`:

```python
@app.route('/new-page')
def new_page():
    return render_template('new_page.html')
```

### Adding New Templates

1. Create a new HTML file in the `templates/` directory
2. Extend the base template:
   ```html
   {% extends "base.html" %}
   
   {% block title %}New Page - Flask App{% endblock %}
   
   {% block content %}
   <h1>New Page Content</h1>
   {% endblock %}
   ```

### Adding Static Files

- **CSS**: Add styles to `static/css/style.css`
- **JavaScript**: Add scripts to `static/js/main.js`
- **Images**: Place images in `static/images/`

## Configuration

The app uses the following configuration:
- **Debug Mode**: Enabled for development
- **Host**: `0.0.0.0` (accessible from any IP)
- **Port**: `5000`
- **Secret Key**: Set via environment variable `SECRET_KEY` or defaults to development key

## Production Deployment

For production deployment:

1. Set the `SECRET_KEY` environment variable
2. Set `FLASK_ENV=production`
3. Use a production WSGI server like Gunicorn
4. Configure a reverse proxy like Nginx

## License

This project is open source and available under the MIT License.
