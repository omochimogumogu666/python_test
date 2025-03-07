import requests
import time
from bs4 import BeautifulSoup
import schedule
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_monitor.log"),
        logging.StreamHandler()
    ]
)

# URL of the product to monitor
PRODUCT_URL = "https://www.pc-koubou.jp/products/detail.php?product_id=1112570&utm_source=affiliate&utm_medium=affiliate&utm_campaign=_Vv6e0WKODg"

# Pushover credentials - you need to get these from Pushover
# Get your user key at https://pushover.net/
PUSHOVER_USER_KEY = "u6hhpwkdm2m4i3mm3fmq3nmygd5jab"
PUSHOVER_API_TOKEN = "ar5b4xb1pbi8f2pe8onbdy7bcdqq68"

# Headers to mimic a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}

def check_stock():
    """Check if the product is in stock at PC-Koubou"""
    try:
        # Send request to the product page
        response = requests.get(PRODUCT_URL, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find stock status element - this selector might need adjustment based on the actual website structure
        stock_status_element = soup.select_one('.item_detail_cart_box .item_detail_cart_box_zaiko')
        
        if stock_status_element:
            stock_status = stock_status_element.text.strip()
            logging.info(f"Current stock status: {stock_status}")
            
            # Check if the product is in stock
            # This condition might need adjustment based on how PC-Koubou indicates stock availability
            if "在庫あり" in stock_status or "在庫残少" in stock_status:
                product_name = soup.select_one('.item_detail_ttl h2').text.strip() if soup.select_one('.item_detail_ttl h2') else "商品"
                price = soup.select_one('.item_detail_price strong').text.strip() if soup.select_one('.item_detail_price strong') else "価格不明"
                
                message = f"在庫復活！\n商品名: {product_name}\n価格: {price}\n{PRODUCT_URL}"
                
                # Send notifications through both LINE and Pushover
                send_pushover_notification(f"在庫復活！", f"商品名: {product_name}\n価格: {price}", PRODUCT_URL)
                
                return True
            else:
                logging.info("Product is still out of stock")
                return False
        else:
            logging.warning("Could not find stock status element on the page")
            return False
    
    except Exception as e:
        logging.error(f"Error checking stock: {e}")
        return False

def send_pushover_notification(title, message, url=None):
    """Send a push notification via Pushover"""
    try:
        # Pushoverのエンドポイント
        pushover_url = "https://api.pushover.net/1/messages.json"
        
        # リクエストデータの準備
        data = {
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": message,
            "title": title,
            "priority": 1,  # 1 = high priority
            "sound": "tugboat"  # You can choose different sounds
        }
        
        # URLが指定されている場合は追加
        if url:
            data["url"] = url
            data["url_title"] = "商品ページを開く"
        
        # POSTリクエストを送信
        response = requests.post(pushover_url, data=data)
        response.raise_for_status()
        
        logging.info("Pushover notification sent successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error sending Pushover notification: {e}")
        return False

def job():
    """Main job to check stock and send notification"""
    logging.info("Running stock check...")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current time: {current_time}")
    
    if check_stock():
        logging.info("Product is in stock! Notification sent.")
    else:
        logging.info("Product is not in stock yet. Will check again later.")

def main():
    """Main function to run the stock monitoring service"""
    logging.info("Starting PC-Koubou stock monitoring service")
    
    # Run the job immediately once
    job()
    
    # Schedule the job to run every 5 minutes
    schedule.every(5).minutes.do(job)
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stock monitoring service stopped by user")

if __name__ == "__main__":
    main()
