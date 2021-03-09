from selenium import webdriver
#from webdriver_manager.chrome import ChromeDriverManager


def download_data_from_sazka(directory='/tmp'):
    #driver = webdriver.Chrome(ChromeDriverManager().install())
    #driver.get("https://www.google.com/")

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1200x3600')
    prefs = {"download.default_directory" : "/tmp"}
    options.add_experimental_option("prefs",prefs)

    driver = webdriver.Chrome(chrome_options=options)
    driver.implicitly_wait(10)

    driver.get('https://www.sazka.cz/loterie/sportka/statistiky')
    element = driver.find_element_by_id('p_lt_ctl07_wPL_p_lt_ctl04_wP_p_lt_ctl01_wS_csvHistory_btnGetCSV')
    driver.get_screenshot_as_file('sazka.png')
    element.click()
    driver.get_screenshot_as_file('click.png')


