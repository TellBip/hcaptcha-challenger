import os
import sys
import time
import uuid
import json
import random
import logging
import asyncio
import argparse
from quart import Quart, request, jsonify
from playwright.async_api import async_playwright

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse
from hcaptcha_challenger.utils import SiteKey

# Logging setup
COLORS = {
    'MAGENTA': '\033[35m',
    'BLUE': '\033[34m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'RED': '\033[31m',
    'RESET': '\033[0m',
}

class CustomLogger(logging.Logger):
    @staticmethod
    def format_message(level, color, message):
        timestamp = time.strftime('%H:%M:%S')
        return f"[{timestamp}] [{COLORS.get(color)}{level}{COLORS.get('RESET')}] -> {message}"

    def debug(self, message, *args, **kwargs):
        super().debug(self.format_message('DEBUG', 'MAGENTA', message), *args, **kwargs)

    def info(self, message, *args, **kwargs):
        super().info(self.format_message('INFO', 'BLUE', message), *args, **kwargs)

    def success(self, message, *args, **kwargs):
        super().info(self.format_message('SUCCESS', 'GREEN', message), *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        super().warning(self.format_message('WARNING', 'YELLOW', message), *args, **kwargs)

    def error(self, message, *args, **kwargs):
        super().error(self.format_message('ERROR', 'RED', message), *args, **kwargs)

logging.setLoggerClass(CustomLogger)
logger = logging.getLogger("HCaptchaAPI")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# Create Quart application
app = Quart(__name__)

# Global variables and parameters
browser_pool = asyncio.Queue()
results = {}
debug_mode = False
api_key = None 
proxy_support = False

# Load results from file
def load_results():
    try:
        if os.path.exists("hcaptcha_results.json"):
            with open("hcaptcha_results.json", "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading results: {str(e)}. Starting with an empty results dictionary.")
    return {}

# Save results to file
def save_results():
    try:
        with open("hcaptcha_results.json", "w") as result_file:
            json.dump(results, result_file, indent=4)
    except IOError as e:
        logger.error(f"Error saving results to file: {str(e)}")

# Initialize browsers
async def initialize_browsers(headless, useragent, thread_count):
    browser_args = []
    if useragent:
        browser_args.append(f"--user-agent={useragent}")
        
    playwright = await async_playwright().start()

    for i in range(thread_count):
        browser = await playwright.chromium.launch(
            headless=headless,
            args=browser_args
        )
        await browser_pool.put((i+1, browser))

        if debug_mode:
            logger.success(f"Browser {i + 1} initialized successfully")

    logger.success(f"Browser pool initialized with {browser_pool.qsize()} browsers")

# Solve captcha
async def solve_hcaptcha(task_id, sitekey, proxy=None):
    index, browser = await browser_pool.get()

    try:
        # Set up context with proxy if specified
        context_options = {}
        if proxy_support and proxy:
            parts = proxy.split(':')
            if len(parts) == 3:
                context_options["proxy"] = {"server": f"{proxy}"}
            elif len(parts) == 5:
                proxy_scheme, proxy_ip, proxy_port, proxy_user, proxy_pass = parts
                context_options["proxy"] = {
                    "server": f"{proxy_scheme}://{proxy_ip}:{proxy_port}", 
                    "username": proxy_user, 
                    "password": proxy_pass
                }
            
        context = await browser.new_context(**context_options)
        page = await context.new_page()

        start_time = time.time()

        if debug_mode:
            logger.debug(f"Browser {index}: Starting hCaptcha solve with Sitekey: {sitekey} | Proxy: {proxy}")
        
        # Create page with hCaptcha
        url = SiteKey.as_site_link(sitekey)
        await page.goto(url)

        # Initialize agent and solve captcha
        agent_config = AgentConfig(GEMINI_API_KEY=api_key)
        agent = AgentV(page=page, agent_config=agent_config)
        
        # Click checkbox and wait for challenge to appear
        await agent.robotic_arm.click_checkbox()
        await agent.wait_for_challenge()
        
        # Check result
        if agent.cr_list:
            cr = agent.cr_list[-1]
            token = cr.generated_pass_UUID
            elapsed_time = round(time.time() - start_time, 3)
            
            logger.success(f"Browser {index}: Successfully solved captcha - {COLORS.get('MAGENTA')}{token[:10]}{COLORS.get('RESET')} in {COLORS.get('GREEN')}{elapsed_time}{COLORS.get('RESET')} Seconds")
            
            results[task_id] = {"token": token, "elapsed_time": elapsed_time}
            save_results()
        else:
            elapsed_time = round(time.time() - start_time, 3)
            results[task_id] = {"token": "CAPTCHA_FAIL", "elapsed_time": elapsed_time}
            if debug_mode:
                logger.error(f"Browser {index}: Error solving hCaptcha in {COLORS.get('RED')}{elapsed_time}{COLORS.get('RESET')} Seconds")
    except Exception as e:
        elapsed_time = round(time.time() - start_time, 3)
        results[task_id] = {"token": "CAPTCHA_FAIL", "elapsed_time": elapsed_time, "error": str(e)}
        if debug_mode:
            logger.error(f"Browser {index}: Error solving hCaptcha: {str(e)}")
    finally:
        if debug_mode:
            logger.debug(f"Browser {index}: Clearing page state")
        
        await context.close()
        await browser_pool.put((index, browser))

# Routes
@app.route('/')
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>hCaptcha Solver API</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-200 min-h-screen flex items-center justify-center">
        <div class="bg-gray-800 p-8 rounded-lg shadow-md max-w-2xl w-full border border-green-500">
            <h1 class="text-3xl font-bold mb-6 text-center text-green-500">Welcome to hCaptcha Solver API</h1>

            <p class="mb-4 text-gray-300">To use the hCaptcha service, send a GET request to 
                <code class="bg-green-700 text-white px-2 py-1 rounded">/hcaptcha</code> with the following query parameters:</p>

            <ul class="list-disc pl-6 mb-6 text-gray-300">
                <li><strong>sitekey</strong>: The site key for hCaptcha</li>
                <li><strong>proxy</strong> (optional): Proxy in format ip:port or scheme://ip:port:user:pass</li>
            </ul>

            <div class="bg-gray-700 p-4 rounded-lg mb-6 border border-green-500">
                <p class="font-semibold mb-2 text-green-400">Example usage:</p>
                <code class="text-sm break-all text-green-300">/hcaptcha?sitekey=914e63b4-ac20-4c24-bc92-cdb6950ccfde</code>
            </div>

            <div class="bg-green-900 border-l-4 border-green-600 p-4 mb-6">
                <p class="text-green-200 font-semibold">API for the hCaptcha Challenger project</p>
                <p class="text-green-300">
                    <a href="https://github.com/QIN2DIM/hcaptcha-challenger" class="text-green-300 hover:underline">GitHub Repository</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/hcaptcha', methods=['GET'])
async def process_hcaptcha():
    """Handle the /hcaptcha endpoint requests."""
    sitekey = request.args.get('sitekey')
    proxy = request.args.get('proxy')

    if not sitekey:
        return jsonify({
            "status": "error",
            "error": "'sitekey' is required"
        }), 400

    task_id = str(uuid.uuid4())
    results[task_id] = "CAPTCHA_NOT_READY"

    try:
        asyncio.create_task(solve_hcaptcha(task_id=task_id, sitekey=sitekey, proxy=proxy))

        if debug_mode:
            logger.debug(f"Request completed with taskid {task_id}.")
        return jsonify({"task_id": task_id}), 202
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/result', methods=['GET'])
async def get_result():
    """Return solved data"""
    task_id = request.args.get('id')

    if not task_id or task_id not in results:
        return jsonify({"status": "error", "error": "Invalid task ID/Request parameter"}), 400

    result = results[task_id]
    
    # If result is not ready yet
    if result == "CAPTCHA_NOT_READY":
        return jsonify({"status": "processing"}), 202
        
    status_code = 200
    if result.get("token") == "CAPTCHA_FAIL":
        status_code = 422

    return jsonify(result), status_code

@app.before_serving
async def startup():
    """Initialization when server starts"""
    global results
    results = load_results()
    
    # Initialize browsers
    await initialize_browsers(
        headless=args.headless, 
        useragent=args.useragent, 
        thread_count=args.thread
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hCaptcha API Server")

    parser.add_argument('--headless', type=bool, default=False, help='Run the browser in headless mode (default: False)')
    parser.add_argument('--useragent', type=str, default=None, help='Specify a custom User-Agent string for the browser')
    parser.add_argument('--debug', type=bool, default=False, help='Enable or disable debug mode (default: False)')
    parser.add_argument('--thread', type=int, default=1, help='Set the number of browser threads (default: 1)')
    parser.add_argument('--proxy', type=bool, default=False, help='Enable proxy support (default: False)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Specify the IP address for the API (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5050, help='Set the port for the API (default: 8080)')
    parser.add_argument('--api_key', type=str, required=True, help='Gemini API key for hCaptcha solving')
    
    args = parser.parse_args()
    
    # Set global parameters
    debug_mode = args.debug
    api_key = args.api_key
    proxy_support = args.proxy
    
    if args.headless is True and args.useragent is None:
        logger.error(f"It is recommended to specify a {COLORS.get('YELLOW')}User-Agent{COLORS.get('RESET')} when using headless mode")
    else:
        app.run(host=args.host, port=args.port, debug=args.debug) 