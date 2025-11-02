from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

def download_data_from_sazka(directory='/tmp'):
    """
    Download lottery data from Sazka website.
    
    Requires Chrome and ChromeDriver to be installed.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--window-size=1200x3600')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    prefs = {"download.default_directory": directory}
    options.add_experimental_option("prefs", prefs)

    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException as e:
        print(f"Error: Could not start Chrome WebDriver")
        print(f"Please ensure Chrome and ChromeDriver are installed:")
        print(f"  - Chrome: https://www.google.com/chrome/")
        print(f"  - ChromeDriver: https://chromedriver.chromium.org/")
        print(f"\nAlternatively, download data manually from:")
        print(f"  https://www.sazka.cz/loterie/sportka/statistiky")
        raise
    
    try:
        driver.implicitly_wait(10)

        driver.get('https://www.sazka.cz/loterie/sportka/statistiky')
        element = driver.find_element(By.ID, 'p_lt_ctl09_wPL_p_lt_ctl03_wS_csvHistory_btnGetCSV')
        driver.get_screenshot_as_file('sazka.png')
        element.click()
        driver.get_screenshot_as_file('click.png')
        driver.quit()
    except Exception as e:
        print(f"Warning: Could not download data from Sazka: {e}")
        print("Please download manually from https://www.sazka.cz/loterie/sportka/statistiky")
        if 'driver' in locals():
            driver.quit()
        raise
