{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


.. autoclass:: {{ objname }}
   :no-undoc-members:

   {% block examples%}
   .. include:: ../gen_modules/backreferences/{{fullname}}.examples
   .. raw:: html

       <div class="sphx-glr-clear"></div>
   {% endblock %}
   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}



