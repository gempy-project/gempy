import os
import requests
import json
import datetime
try:
    import pyqrcode as qr
    PYQRCODE_IMPORT = True
except ImportError:
    PYQRCODE_IMPORT = False

package_directory = os.path.dirname(os.path.abspath(__file__))


class RexAPI:

    def __init__(self, project_name, api_token=None, secret=None):
        """Rest client to connect to the RexOS system


        Args:
            project_name:
            api_token:
            secret:
        """
        self.response = None  # saves the most current server response

        self.token_ID, self.secret = self.load_credentials(api_token=api_token, secret=secret)
        self.access_token = self.authorize_session()
        self.owner = self.get_user_information()
        self.project_name = project_name
        self.project_urn, self.project_link = self.create_project(self.project_name)
        self.root_reference_link, self.root_reference_key = self.create_root_reference()
        self.file_reference_link = self.create_file_resource_reference()

        self.project_file_urn = None
        self.project_file_uplink = None
        self.project_reference = None

        self.rextag = None

    def load_credentials(self, filename=os.path.join(package_directory, 'RexCloud_Api_key.txt'),
                         api_token: str = None, secret: str = None):

        if not os.path.isfile(filename) or (api_token is not None and secret is not None):
            file = open(filename, 'w')
            login_data = False
            if api_token is None or secret is None:
                login_data = True

            if api_token is None:
                api_token = 'REPLACE_TEXT_WITH_YOUR_API_Token'
            if secret is None:
                secret = 'REPLACE_TEXT_WITH_SECRET'

            file.write(api_token+'\n'+secret+'\n'+' # put your API tokens and secrets in the lines above.\n'
                                                  ' # Do not track the file on git.')
            file.close()

            if login_data:
                raise AttributeError('Cache key is not created. You need to pass as argument the REX api_token'
                                     ' and secret, or adding them in RexCloud_API_key.txt.'
                                     ' https://www.rexos.org/getting-started/')

        with open(filename, "r") as credential_file:
            token_id = credential_file.readline().strip('\n')
            secret = credential_file.readline().strip('\n')

        return token_id, secret

    def authorize_session(self):

        headers = {'Accept': 'application/json;charset=UTF-8',
                   'Content-Type': 'application/x-www-form-urlencoded; charset=ISO-8859-1'}

        data = {'grant_type': 'client_credentials'}

        self.response = requests.post('https://rex.robotic-eyes.com/oauth/token',
                                      headers=headers, data=data, auth=(self.token_ID, self.secret)
                                      )
        if self.response.status_code == 200:
            access_token = self.response.json()['access_token']
            return access_token

        else:
            raise ConnectionError("something went wrong! Status code: "+str(self.response.status_code) +
                                  'Probably the token or the secret is not valid. '
                                  'https://www.rexos.org/getting-started/')

    def get_user_information(self):
        headers = {'Authorization': 'Bearer ' + self.access_token, 'Accept': 'application/json;charset=UTF-8'}
        self.response = requests.get('https://rex.robotic-eyes.com/api/v2/users/current', headers=headers)

        if self.response.status_code == 200:
            owner = self.response.json()['userId']
            return owner

        else:
            print("something went wrong! Status code: "+str(self.response.status_code))

    def create_project(self, project_name):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8'
        }

        data = json.dumps({"name" : project_name,  "owner" : self.owner})  # this call expects json!

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/projects', headers=headers, data=data)

        if self.response.status_code == 201:
            project_urn = self.response.json()['urn']
            project_link = self.response.json()['_links']['self']['href']

            return project_urn, project_link

        else:
            raise ConnectionError("something went wrong! Status code: " + str(self.response.status_code) +
                                  'Probably project name already exists.')

    def create_root_reference(self):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8',
        }

        data = json.dumps({"project" : self.project_link,
                "name" : "root reference",
                "address" : {"addressLine1": "Sample", "postcode": "52072",
                             "city": "Aachen",   "country" : "Austria"}}
                          )

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/rexReferences', headers=headers, data=data)
        if self.response.status_code == 201:
            root_reference_link = self.response.json()['_links']['self']['href']
            root_reference_key = self.response.json()['key']

            return root_reference_link, root_reference_key

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

    def create_file_resource_reference(self):
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8',
        }

        data = json.dumps({"project" : self.project_link,
                           "name" : "file ressource reference",
                           "rootReference" : "false",  #setting root reference to false
                           "parentReference" : self.root_reference_link})

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/rexReferences', headers=headers, data=data)
        if self.response.status_code == 201:
            file_reference_link = self.response.json()['_links']['self']['href']

            return file_reference_link

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

    def create_project_file(self, projectname):
        headers = {
            'Authorization':'Bearer ' + self.access_token,
            'Accept': 'application/json;charset=UTF-8',
            'Content-Type': 'application/json;charset=UTF-8',
                 }

        data = json.dumps({"project" : self.project_link,
                           "name" : projectname,
                           "type" : "rex",

                           "rexReference" : self.file_reference_link
                           })

        self.response = requests.post('https://rex.robotic-eyes.com/api/v2/projectFiles', headers=headers, data=data)

        if self.response.status_code == 201:
            self.project_file_urn = self.response.json()['urn']
            self.project_file_uplink = self.response.json()['_links']['file.upload']['href']
            self.project_reference = self.response.json()['rexReferenceKey']

            return True

        else:
            print("something went wrong! Status code: " + str(self.response.status_code))

            return False

    def upload_rexfile(self, filename):

        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'contentType': 'application/octet-stream',
            'type' : 'rex'
        }

        file = {
            'file':  (filename, open(filename, 'rb')),

        }

        self.response = requests.post(self.project_file_uplink,
                                 headers=headers, files=file)

    def return_rextag(self):
        self.rextag = Rextag(self.project_reference)

        return self.rextag


class Rextag:

    def __init__(self, project_reference):
        self.rextag_url, self.rextag = self.create_rextag(project_reference)

    def __repr__(self):
        return self.rextag.terminal(module_color="reverse", background="default", quiet_zone=1)

    def create_rextag(self, project_reference):
        if PYQRCODE_IMPORT is False:
            raise ImportError('This method depends on pyqrcode and it is not possible to import.')
        base_url = "https://rex.codes/v1/"
        rextag_url = base_url+project_reference
        rextag = qr.create(rextag_url)
        return rextag_url, rextag

    def display_tag(self, reverse=True):
        """
        displays the rextag to the terminal standard output as ascii.

        you can invert the color by setting inverse=True, this is necessary if you run it in a jupyter notebook
        after creation, you can save the rextag as svg using the rextag.svg method:
        self.rextag.svg("project_name", scale=8)

        Args:
            reverse: (boolean) inverts background and foreground color in the terminal output

        Returns: None, prints a QR code to terminal output

        """

        if reverse:
            print(self.rextag.terminal(module_color="reverse", background="default", quiet_zone=1))

        else:
            print(self.rextag.terminal(quiet_zone=1))

        return True

    def save_svg(self, filename):
        self.rextag.svg(filename, scale=8)


def upload_to_rexcloud(infiles : list, project_name=None, **kwargs):
    """
    wrapper around api calls to upload rexfiles of a gempy model.

    you will need to register an account under https://app.rexos.cloud/ .
    create an api key and store the key and secret in RexCloud_Api_key.txt.

    the function will take a list of rexos input filenames and uploads them into a newly created project.

    an ar code is plotted that can be scanned with the rexview app to show the model in vr.
    the qr code is plotted in ascii to the standard output, if the qr code is not recognizable,
     try to revert the colors o0f the terminal output by setting reverse=False in thew show_tag() Method.

    all api calls are python implementation of the Rex os api:
    https://www.rexos.org/rex-api/#tutorial-rex-project-before-you-begin


    Args:
        infiles: List of rexos file names
        project_name: name of the project under which it appears in the rexcloud. if none is specified,
        the current timestamp is used.

    Returns:
        A Rextag Object

    """

    if project_name is None:
        timestamp = datetime.datetime.now()
        project_name = str(timestamp)

    api = RexAPI(project_name, **kwargs)

    for file in infiles:

        api.create_project_file(file)
        api.upload_rexfile(file)

    tag = api.return_rextag()

    return tag


