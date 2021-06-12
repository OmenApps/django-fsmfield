===============
django-fsmfield
===============


.. image:: https://img.shields.io/pypi/v/fsmfield.svg
        :target: https://pypi.python.org/pypi/fsmfield

.. image:: https://img.shields.io/travis/dryprojects/fsmfield.svg
        :target: https://travis-ci.com/dryprojects/fsmfield

.. image:: https://readthedocs.org/projects/fsmfield/badge/?version=latest
        :target: https://fsmfield.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




finite state machine for django


* Free software: MIT license


Installation
-------------------

.. code-block:: sh

   $ pip install django-fsmfield

or

.. code-block:: sh

   $ python setup install

Usage
-------------------

.. code-block:: python

    from fsmfield import FSMField, FSMMixin


    class MyStatus:
        # fsm events
        SET_SUCCESS = 'set_success'
        SET_FAILED = 'set_failed'
        SET_TIMEOUT = 'set_timeout'
        SET_DOING = 'set_doing'
        SET_PENDING = 'set_pending'

        # basic states
        SUCCESS = 'success'
        FAILED = 'failed'
        TIMEOUT = 'timeout'
        RUNNING = 'doing'
        PENDING = 'pending'

        DONE_STATES = [
            SUCCESS,
            FAILED,
            TIMEOUT
        ]

        FSM_STATES = [
            SUCCESS,
            FAILED,
            TIMEOUT,
            RUNNING,
            PENDING
        ]

        FSM_INITIAL_STATE = PENDING

        STATE_CHOICES = list(zip(FSM_STATES, FSM_STATES))

        FSM_TRANSITIONS = [
            {
                "trigger": SET_SUCCESS,
                "source": RUNNING,
                "dest": SUCCESS
            },
            {
                "trigger": SET_FAILED,
                "source": RUNNING,
                "dest": FAILED,
                "after": "after_state_failed"
            },
            {
                "trigger": SET_TIMEOUT,
                "source": RUNNING,
                "dest": TIMEOUT,
                "after": "after_state_timeout"
            },
            {
                "trigger": SET_DOING,
                "source": PENDING,
                "dest": RUNNING,
                "after": "after_state_running"
            },
            {
                "trigger": SET_PENDING,
                "source": [FAILED, TIMEOUT, RUNNING],
                "dest": PENDING
            }
        ]


    class MyModel(FSMMixin, models.Model):
        state = FSMField(
            verbose_name="fsm state",
            max_length=20,
            transitions=MyStatus.FSM_TRANSITIONS,
            choices=MyStatus.STATE_CHOICES,
            default=MyStatus.FSM_INITIAL_STATE,
            after_state_change='after_state_change',
            send_event=True,
        )

        def after_state_change(self, event):
            self.save(update_fields=('state',))

        def after_state_running(self, event): ...
        def after_state_timeout(self, event): ...
        def after_state_failed(self, event): ...

    >>> obj = MyModel.objects.create()
    >>> obj.is_pending() # True
    >>> obj.set_doing() # enter state doing
    >>> obj.set_success()

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
