import os
import requests
package_directory = os.path.dirname(os.path.abspath(__file__))


class RexAPI:

    def __init__(self):
        self.api_key = None
        self.token_ID = None
        self.secret = None
        self.response = None  # saves the most current server response for debugging

    def read_credentials(self, filename = os.path.join(package_directory, 'RexCloud_Api_key.txt')):
        with open(filename, "r") as file:
            self.token_ID = file.readline().strip('\n')
            self.secret = file.readline().strip('\n')

    def authorize_session(self):

        headers = {'Accept': 'application/json;charset=UTF-8',
                   'Content-Type': 'application/x-www-form-urlencoded; charset=ISO-8859-1'}

        data = {'grant_type': 'client_credentials'}

        self.response = requests.post('https://rex.robotic-eyes.com/oauth/token',
                                      headers=headers, data=data, auth=(self.token_ID, self.secret)
                                      )
        if self.response.status_code == 200:
            self.access_token = self.response.json()['access_token']

        else:
            print("something went wrong! Status code: "+str (self.response.status_code))
