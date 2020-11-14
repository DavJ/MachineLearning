from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_data_from_sazka(directory='./downloads'):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1200x3600')
    prefs = {"download.default_directory" : directory}
    options.add_experimental_option("prefs",prefs)

    driver = webdriver.Chrome(chrome_options=options)
    #driver.implicitly_wait(25)
    driver.get('https://www.sazka.cz/loterie/sportka/statistiky')
    try:
        element = WebDriverWait(driver, 10).until(
           EC.element_to_be_clickable((By.ID, "p_lt_ctl08_wPL_p_lt_ctl04_wP_p_lt_ctl01_wS_csvHistory_btnGetCSV"))
        )
        driver.get_screenshot_as_file('sazka.png')
        element.click()
        driver.get_screenshot_as_file('sazka.png')   	
    finally:
        driver.quit()

if __name__ == "__main__":
 download_data_from_sazka()

