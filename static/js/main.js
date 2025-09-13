// Main JavaScript file for Flask app

// Test API function
function testAPI() {
    const responseDiv = document.getElementById('api-response');
    const contentDiv = document.getElementById('api-response-content');
    
    // Show loading state
    responseDiv.style.display = 'block';
    responseDiv.className = 'alert alert-info';
    contentDiv.textContent = 'Loading...';
    
    // Make API call
    fetch('/api/hello')
        .then(response => response.json())
        .then(data => {
            responseDiv.className = 'alert alert-success';
            contentDiv.textContent = JSON.stringify(data, null, 2);
        })
        .catch(error => {
            responseDiv.className = 'alert alert-danger';
            contentDiv.textContent = 'Error: ' + error.message;
        });
}

// Test echo API function
function testEchoAPI() {
    const testData = {
        message: 'Hello from JavaScript!',
        timestamp: new Date().toISOString(),
        user: 'Flask App User'
    };
    
    const responseDiv = document.getElementById('api-response');
    const contentDiv = document.getElementById('api-response-content');
    
    // Show loading state
    responseDiv.style.display = 'block';
    responseDiv.className = 'alert alert-info';
    contentDiv.textContent = 'Sending data...';
    
    // Make API call
    fetch('/api/echo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(testData)
    })
    .then(response => response.json())
    .then(data => {
        responseDiv.className = 'alert alert-success';
        contentDiv.textContent = JSON.stringify(data, null, 2);
    })
    .catch(error => {
        responseDiv.className = 'alert alert-danger';
        contentDiv.textContent = 'Error: ' + error.message;
    });
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    console.log('Flask app loaded successfully!');
    
    // Add any initialization code here
    const currentYear = new Date().getFullYear();
    const footerYear = document.querySelector('footer p');
    if (footerYear) {
        footerYear.textContent = footerYear.textContent.replace('2024', currentYear);
    }
});
