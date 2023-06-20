class MetaData(object):
    """Class containing metadata of the project.

    Set of attributes and methods that are not related directly to the geological model but more with the project

    Args:
        project_name (str): Name of the project. This is used as the default value for some I/O actions

    Attributes:
        date (str): Time of the creation of the project
        project_name (str): Name of the project. This is used as the default value for some I/O actions

    """

    def __init__(self, project_name='default_project'):
        import datetime
        now = datetime.datetime.now()
        self.date = now.strftime(" %Y-%m-%d %H:%M")

        if project_name == 'default_project':
            project_name += self.date

        self.project_name = project_name
