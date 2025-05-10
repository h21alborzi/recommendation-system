from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time

import pandas as pd

# Setup
path = "C:\\Users\\ha\\Desktop\\chromedriver\\chromedriver.exe"
options = Options()

data = []

# options.add_argument("--headless")  # Uncomment to run headless

service = Service(executable_path=path)
driver = webdriver.Chrome(service=service, options=options)

try:
    p = 1
    while True:
        website = f"https://printify.com/app/products/accessories/phone-cases?page={p}"
        driver.get(website)
        time.sleep(10)  # Wait for page to load. You can use WebDriverWait here instead.

        print(f"\n--- Page {p} ---")

        name_elements = driver.find_elements(By.XPATH, '//p[@data-testid="blueprintName"]')

        # image_elements = driver.find_elements(By.XPATH, '//img[@data-testid="hoverImage"]')

        for i, element in enumerate(name_elements):
            print(f"Item {i+1}: {element.text}")

            data.append({"name": element.text, "number": f"{p}:{i+1}"})
        break
        # for i, element in enumerate(image_elements):
        #     print(f"image {i+1}: {element.text}")

        # Find pagination
        # current_button = driver.find_element(By.XPATH, '//a[@class="page-number active"]')
        # next_buttons = driver.find_elements(By.XPATH, '//a[@class="page-number"]')

        # next_found = False
        # for btn in next_buttons:
        #     try:
        #         if int(btn.text) - int(current_button.text) == 1:
        #             p += 1
        #             next_found = True
        #             break
        #     except ValueError:
        #         continue  # Sometimes there might be non-numeric buttons (like "…" or "Next")

        # if not next_found:
        #     break  # No next page

finally:
    driver.quit()


df = pd.DataFrame(data)
df.to_csv("printify_phonecase.csv", index=False)