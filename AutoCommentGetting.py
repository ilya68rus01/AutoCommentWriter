from selenium import webdriver
from selenium.common.exceptions import *
import numpy as np
import codecs
import time
import pandas as pd

class AutoCommentGetting:
    def __init__(self):
        self.driver = webdriver.Chrome()


    def f(self, l):
        n = []
        for i in l:
            if i not in n:
                n.append(i)
        return n

    def get_storys(self):
        i = 0
        j = 0
        hrefs = list()
        self.driver.get("https://pikabu.ru/best/01-01-2018_28-04-2020")
        time.sleep(3)
        for y in range(0, 345600, 8640):
            self.driver.execute_script("window.scrollTo(0,"+ str(y) +")")
            time.sleep(8)
            storys = self.driver.find_elements_by_class_name("story__title")
            for story in storys:
                link = story.find_elements_by_class_name("story__title-link")
                hrefs.append(link[0].get_attribute("href"))
        hrefs = self.f(hrefs)
        print((np.size(hrefs)))
        print(hrefs)
        index = 0
        for href in hrefs:
            data_frame = pd.DataFrame(columns=['Comment'])
            self.driver.get(href)
            time.sleep(5)
            try:
                while True:
                    btnL = self.driver.find_elements_by_class_name('comments__more-button')
                    if btnL == list():
                        print("empty")
                        break
                    else:
                        for btn in btnL:
                            btn.click()
                            time.sleep(5)
            except (ElementNotInteractableException):
                print("all comments ready on " + href)
            comments = self.driver.find_elements_by_class_name('comment')
            for comment in comments:
                if str(comment.find_element_by_class_name('comment__content').text) != "":
                    data_frame.loc[j] = str(comment.find_element_by_class_name('comment__content').text)
                    j += 1
            i += 1
            # np.savetxt(r'data/np' + str(index) + '.txt', data_frame.values, fmt='%')
            data_frame.to_csv("data/dataframeTest" + str(index) + ".txt", index=False, header=False)
            index += 1

        return []

