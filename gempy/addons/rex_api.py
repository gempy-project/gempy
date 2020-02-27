import os
import requests
import json
package_directory = os.path.dirname(os.path.abspath(__file__))


class RexAPI:

    def __init__(self):
        self.api_key = None
        self.token_ID = None
        self.secret = None
        self.owner = None
        self.access_token = None
        self.project_urn = None
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

    def get_user_information(self):
        headers = {'Authorization': 'Bearer ' + self.access_token, 'Accept': 'application/json;charset=UTF-8'}
        self.response = requests.get('https://rex.robotic-eyes.com/api/v2/users/current', headers=headers)

        if self.response.status_code == 200:
            self.owner = self.response.json()['userId']

        else:
            print("something went wrong! Status code: "+str (self.response.status_code))

    def create_project(self, project_name):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8'
        }

        data = json.dumps({"name" : project_name,  "owner" : self.owner})  # this call expects json!

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/projects', headers=headers, data=data)

        if self.response.status_code == 201:
            self.project_urn = self.response.json()['urn']

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

    def retrieve_project(self): #this is necessary to get an upload link for the rexfiles
        headers = {'Authorization': 'Bearer ' + self.access_token,
                    'Accept': 'application/json;charset=UTF-8'}

        self.response = requests.get('http://curl', headers=headers)

    def upload_rexfile(self,filename):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
          #  'Content-Type': 'multipart/form-data; boundary="7YHbCQEvZJ4UpDxLWav_05SOJpLdJKI6541wYs6_"',
        }

        files = {
            'file': open(filename,"r") , 'type': 'rex'
        }

        response = requests.post('http://curl', headers=headers, files=files)

