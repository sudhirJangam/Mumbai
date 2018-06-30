#!/usr/bin/python

import urllib
import urllib2
import cookielib
import BeautifulSoup

url = "https://accounts.google.com/ServiceLogin?hl=en";
values = {'Email': 'sudhir.sj@gmail.com', 'Passwd' : 'Sanvi123', 'signIn' : 'Sign in', 'PersistentCookie' : 'yes'} # The form data 'name' : 'value'

cookie = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie))
data = urllib.urlencode(values)
response = self.opener.open(url, data)
print (response)