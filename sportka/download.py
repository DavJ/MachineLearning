from selenium import webdriver
from selenium.webdriver.common.by import By

def download_data_from_sazka(directory='/tmp'):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--window-size=1200x3600')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    prefs = {"download.default_directory": directory}
    options.add_experimental_option("prefs", prefs)

    try:
        driver = webdriver.Chrome(options=options)
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
        raise
