#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

import enum
import inspect
import itertools
import logging
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Optional, Callable, Union, List

from django.db.models.fields import CharField
from django.utils.deconstruct import deconstructible

from .exceptions import FSMError

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def listvalue(value):
    if value is None:
        return []
    try:
        return value if isinstance(value, (list, tuple)) else [value]
    except ReferenceError:
        return [value]


class EventData:
    def __init__(self, state, event, machine, model, args, kwargs):
        self.state = state
        self.event = event
        self.machine = machine
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.transition = None
        self.error = None
        self.result = None

    def update(self, state):
        if not isinstance(state, State):
            self.state = self.machine.get_state(state)
        else:
            self.state = state

    def __str__(self):
        return f"<{type(self).__name__}({self.state!r}, {getattr(self, 'transition')!r})@{id(self)}>"


class Event:
    def __init__(self, name, machine):
        self.name = name
        self.machine = machine
        self.transitions = defaultdict(list)

    def add_transition(self, transition):
        self.transitions[transition.source].append(transition)

    def trigger(self, model, *args, **kwargs):
        func = partial(self._trigger, model, *args, **kwargs)
        # pylint: disable=protected-access
        return self.machine._process(func)

    def _trigger(self, model, *args, **kwargs):
        state = self.machine.get_model_state(model)
        if state.name not in self.transitions:
            msg = f"{self.machine.label!r} Can't trigger event {self.name!r} from state {state.name!r}!"
            ignore = state.ignore_invalid_triggers if state.ignore_invalid_triggers is not None \
                else self.machine.ignore_invalid_triggers
            if ignore:
                _LOGGER.warning(msg)
                return False
            else:
                raise FSMError(msg)
        event_data = EventData(state, self, self.machine, model, args, kwargs)
        return self._process(event_data)

    def _process(self, event_data):
        self.machine.callbacks(self.machine.prepare_event, event_data)
        _LOGGER.debug(f"{self.machine.label!r} Executed machine preparation callbacks before conditions.")
        try:
            for trans in self.transitions[event_data.state.name]:
                event_data.transition = trans
                if trans.execute(event_data):
                    event_data.result = True
                    break
        except Exception as err:
            event_data.error = err
            if self.machine.on_exception:
                self.machine.callbacks(self.machine.on_exception, event_data)
            else:
                raise
        finally:
            try:
                self.machine.callbacks(self.machine.finalized_event, event_data)
                _LOGGER.debug(f"{self.machine.label!r} Executed machine finalize callbacks")
            except Exception as err:
                _LOGGER.error(f"{self.machine.label!r} While executing finalize callbacks a"
                              f" {type(err).__name__!r} occurred: {str(err)}.")

        return event_data.result

    def __str__(self):
        return f"<{type(self).__name__!r}({self.name!r})@{id(self)}>"

    def add_callback(self, trigger, func):
        for trans in itertools.chain(*self.transitions.values()):
            trans.add_callback(trigger, func)


@deconstructible
class State:
    dynamic_callbacks = ['on_enter', 'on_exit']

    def __init__(
        self,
        name: Union[str, enum.Enum],
        on_enter: Optional[Union[Callable[[EventData], bool], List[Callable[[EventData], bool]], str]] = None,
        on_exit: Optional[Union[Callable[[EventData], bool], List[Callable[[EventData], bool]], str]] = None,
        ignore_invalid_triggers=False
    ):
        """
        :param name: The name of the state
        :param on_enter: Optional callable(s) to trigger when a state is entered. Can be either a string providing
        the name of a callable, or a list of strings.
        :param on_exit: Optional callable(s) to trigger when a state is exited. Can be either a string providing
        the name of a callable, or a list of strings.
        :param ignore_invalid_triggers: Optional flag to indicate if unhandled/invalid triggers should raise an
        exception
        """
        self._name = name
        self.ignore_invalid_triggers = ignore_invalid_triggers
        self.on_enter = listvalue(on_enter) if on_enter else []
        self.on_exit = listvalue(on_exit) if on_exit else []

    @property
    def name(self):
        if isinstance(self._name, enum.Enum):
            return self._name.name

        return self._name

    @property
    def value(self):
        return self._name

    def enter(self, event_data):
        """Triggered when a state is entered."""
        _LOGGER.debug(f"{event_data.machine.label!r} Entering state {self.name!r}.")
        event_data.machine.callbacks(self.on_enter, event_data)
        _LOGGER.debug(f"{event_data.machine.label!r} Finished processing state {self.name!r} enter.")

    def exit(self, event_data):
        """Triggered when a state is exited."""
        _LOGGER.debug(f"{event_data.machine.label!r} Exiting state {self.name!r}")
        event_data.machine.callbacks(self.on_exit, event_data)
        _LOGGER.debug(f"{event_data.machine.label!r} Finished processing state {self.name!r} exit.")

    def add_callback(self, trigger: str, func: Union[Callable[[Event], bool], str]):
        """Add a new enter or exit callback.
        :param trigger: The type of triggering event. Must be one of 'enter' or 'exit'.
        :param func: The name of the callback function.
        """
        getattr(self, f'on_{trigger}').append(func)

    def __str__(self):
        return f"<{type(self).__name__}({self.name!r})@{id(self)}>"


class Condition:
    def __init__(self, func, target=True):
        self.func = func
        self.target = target

    def check(self, event_data):
        predicate = event_data.machine.resolve_callable(self.func, event_data)
        if event_data.machine.send_event:
            return predicate(event_data) == self.target
        return predicate(*event_data.args, **event_data.kwargs) == self.target

    def __str__(self):
        return f"<{type(self).__name__}({self.func!r})@{id(self)}>"


class Transition:
    condition_cls = Condition
    dynamic_callbacks = ['before', 'after', 'prepare']

    def __init__(self, source, dest, conditions=None, unless=None, before=None, after=None, prepare=None):
        self.source = source
        self.dest = dest
        self.prepare = listvalue(prepare) if prepare else []
        self.before = listvalue(before) if before else []
        self.after = listvalue(after) if after else []
        self.conditions = []
        if conditions is not None:
            for cond in listvalue(conditions):
                self.conditions.append(self.condition_cls(cond))
        if unless is not None:
            for cond in listvalue(unless):
                self.conditions.append(self.condition_cls(cond, target=False))

    def _eval_conditions(self, event_data):
        for cond in self.conditions:
            if not cond.check(event_data):
                _LOGGER.debug(
                    f"{event_data.machine.label!r} Transition condition failed:"
                    f" {cond.func!r}() does not return {cond.target!r}. Transition halted.")
                return False
        return True

    def execute(self, event_data):
        _LOGGER.debug(f"{event_data.machine.label!r} Initiating transition from"
                      f" state {self.source!r} to state {self.dest!r}...")
        event_data.machine.callbacks(self.prepare, event_data)
        _LOGGER.debug(f"{event_data.machine.label!r} Executed callbacks before conditions.")
        if not self._eval_conditions(event_data):
            return False

        event_data.machine.callbacks(itertools.chain(
            event_data.machine.before_state_change, self.before
        ), event_data)

        _LOGGER.debug(f"{event_data.machine.label!r} Executed callback before transition.")

        if self.dest:  # if self.dest is None this is an internal transition with no actual state change
            self._change_state(event_data)

        event_data.machine.callbacks(itertools.chain(
            self.after, event_data.machine.after_state_change
        ), event_data)
        _LOGGER.debug(f"{event_data.machine.label!r} Executed callback after transition.")
        return True

    def _change_state(self, event_data):
        event_data.machine.get_state(self.source).exit(event_data)
        event_data.machine.set_model_state(self.dest, event_data.model)
        event_data.update(getattr(event_data.model, event_data.machine.name))
        event_data.machine.get_state(self.dest).enter(event_data)

    def add_callback(self, trigger, func):
        getattr(self, trigger).append(func)

    def __str__(self):
        return f"<{type(self).__name__}({self.source!r}, {self.dest!r})@{id(self)}>"


class FSMField(CharField):
    state_cls = State
    transition_cls = Transition
    event_cls = Event
    state_all_wildcard = "*"
    state_same_wildcard = "="

    @staticmethod
    def resolve_callable(func, event_data):
        """ Converts a model's property name, method name or a path to a callable into a callable.
            If func is not a string it will be returned unaltered.
        Args:
            func (str or callable): Property name, method name or a path to a callable
            event_data (EventData): Currently processed event
        Returns:
            callable function resolved from string or func
        """
        if isinstance(func, str):
            try:
                func = getattr(event_data.model, func)
                if not callable(func):  # if a property or some other not callable attribute was passed
                    def func_wrapper(*_, **__):  # properties cannot process parameters
                        return func

                    return func_wrapper
            except AttributeError:
                try:
                    mod, name = func.rsplit('.', 1)
                    m = __import__(mod)
                    for n in mod.split('.')[1:]:
                        m = getattr(m, n)
                    func = getattr(m, name)
                except (ImportError, AttributeError, ValueError):
                    raise AttributeError(f"Callable with name {func!r} could neither be retrieved from the passed "
                                         f"model nor imported from a module.")
        return func

    @staticmethod
    def _check_model_binding(model, method_name, method_callback):
        if hasattr(model, method_name):
            _LOGGER.warning(f"{model} already contains an attribute {method_name!r}. Skip binding.")
        else:
            setattr(model, method_name, partial(method_callback, model))

    @staticmethod
    def _process(trigger):
        return trigger()

    @classmethod
    def _create_transition(cls, *args, **kwargs):
        return cls.transition_cls(*args, **kwargs)

    @classmethod
    def _create_event(cls, *args, **kwargs):
        return cls.event_cls(*args, **kwargs)

    @classmethod
    def _create_state(cls, *args, **kwargs):
        return cls.state_cls(*args, **kwargs)

    @classmethod
    def _identify_callback(cls, name):
        # Does the prefix match a known callback?
        for callback in itertools.chain(cls.state_cls.dynamic_callbacks, cls.transition_cls.dynamic_callbacks):
            if name.startswith(callback):
                callback_type = callback
                break
        else:
            return None, None

        # Extract the target by cutting the string after the type and separator
        target = name[len(callback_type) + 1:]

        # Make sure there is actually a target to avoid index error and enforce _ as a separator
        if target == '' or name[len(callback_type)] != '_':
            return None, None

        return callback_type, target

    @property
    def before_state_change(self):
        return self._before_state_change

    @before_state_change.setter
    def before_state_change(self, value):
        self._before_state_change = listvalue(value)

    @property
    def after_state_change(self):
        return self._after_state_change

    @after_state_change.setter
    def after_state_change(self, value):
        self._after_state_change = listvalue(value)

    @property
    def prepare_event(self):
        return self._prepare_event

    @prepare_event.setter
    def prepare_event(self, value):
        self._prepare_event = listvalue(value)

    @property
    def finalized_event(self):
        return self._finalized_event

    @finalized_event.setter
    def finalized_event(self, value):
        self._finalized_event = listvalue(value)

    @property
    def on_exception(self):
        return self._on_exception

    @on_exception.setter
    def on_exception(self, value):
        self._on_exception = listvalue(value)

    def __init__(
        self,
        transitions=None,
        send_event=False,
        ignore_invalid_triggers=False,
        before_state_change=None,
        after_state_change=None,
        prepare_event=None,
        finalized_event=None,
        on_exception=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not self.choices:
            raise ValueError(f"'choices' can't be null value.")
        self.label = 'fsm'
        self.states = OrderedDict()
        self.events = {}
        self.ignore_invalid_triggers = ignore_invalid_triggers
        self.add_states([state[0] for state in self.choices])
        self.send_event = send_event
        self._on_exception = listvalue(on_exception)
        self._before_state_change = listvalue(before_state_change)
        self._after_state_change = listvalue(after_state_change)
        self._prepare_event = listvalue(prepare_event)
        self._finalized_event = listvalue(finalized_event)
        if transitions:
            self.add_transitions(transitions)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(f"{name!r} does not exist on <{type(self)}@{id(self)}>")

        # Could be a callback
        callback_type, target = self._identify_callback(name)

        if callback_type is not None:
            if callback_type in self.transition_cls.dynamic_callbacks:
                if target not in self.events:
                    raise AttributeError(f"event {target!r} is not registered on <{type(self)}@{id(self)}>")
                return partial(self.events[target].add_callback, callback_type)

            elif callback_type in self.state_cls.dynamic_callbacks:
                state = self.get_state(target)
                return partial(state.add_callback, callback_type[3:])

        try:
            return self.__getattribute__(name)
        except AttributeError:
            # Nothing matched
            raise AttributeError(f"{name!r} does not exist on <{type(self)}@{id(self)}>")

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.on_exception:
            kwargs['on_exception'] = self.on_exception
        if self.send_event:
            kwargs['send_event'] = self.send_event
        if self.ignore_invalid_triggers:
            kwargs['ignore_invalid_triggers'] = self.ignore_invalid_triggers

        kwargs['on_exception'] = self._on_exception
        kwargs['before_state_change'] = self._before_state_change
        kwargs['after_state_change'] = self._after_state_change
        kwargs['prepare_event'] = self._prepare_event
        kwargs['finalized_event'] = self._finalized_event

        return name, path, args, kwargs

    def init_model(self, model):
        self._set_model_states_bindings(model)
        self._set_triggers_to_model(model)
        self.label = getattr(model, 'fsm_label', self.label)

    def add_transition(
        self,
        trigger,
        source,
        dest,
        conditions=None,
        unless=None,
        before=None,
        after=None,
        prepare=None,
        **kwargs
    ):
        if trigger == self.name:
            raise ValueError("Trigger can't be same as model fsm field name")

        if trigger not in self.events:
            self.events[trigger] = self._create_event(trigger, self)

        if source == self.state_all_wildcard:
            source = list(self.states.keys())
        else:
            source = [
                s.name if isinstance(s, State) and self._has_state(s, raise_error=True) or hasattr(s, 'name') else s for
                s in listvalue(source)]

        for state in source:
            if dest == self.state_same_wildcard:
                _dest = state
            elif dest is not None:
                if isinstance(dest, State):
                    _ = self._has_state(dest, raise_error=True)
                _dest = dest.name if hasattr(dest, 'name') else dest
            else:
                _dest = None
            _trans = self._create_transition(
                state,
                _dest,
                conditions,
                unless,
                before,
                after,
                prepare,
                **kwargs
            )
            self.events[trigger].add_transition(_trans)

    def add_transitions(self, transitions):
        for tans in listvalue(transitions):
            self.add_transition(**tans)

    def get_state(self, state):
        if isinstance(state, enum.Enum):
            state = state.name

        if state not in self.states:
            raise ValueError(f"State {state} is not a registered state.")

        return self.states[state]

    def get_model_state(self, model):
        return self.get_state(getattr(model, self.name))

    def set_model_state(self, state, model):
        if not isinstance(state, State):
            state = self.get_state(state)
        setattr(model, self.name, state.value)

    def add_states(self, states, on_enter=None, on_exit=None, ignore_invalid_triggers=None, **kwargs):
        if not ignore_invalid_triggers:
            ignore_invalid_triggers = self.ignore_invalid_triggers
        states = listvalue(states)
        for state in states:
            if isinstance(state, (str, enum.Enum)):
                state = self._create_state(
                    state,
                    on_enter=on_enter,
                    on_exit=on_exit,
                    ignore_invalid_triggers=ignore_invalid_triggers,
                    **kwargs
                )
            elif isinstance(state, dict):
                if 'ignore_invalid_triggers' not in state:
                    state['ignore_invalid_triggers'] = ignore_invalid_triggers
                state = self._create_state(**state)

            if not isinstance(state, State):
                raise TypeError(f"Invalid FSM State type {type(state)!r}!")

            self.states[state.name] = state

    def is_state(self, state_value, model):
        return getattr(model, self.name) == state_value

    def callbacks(self, funcs, event_data):
        for func in funcs:
            self.callback(func, event_data)
            _LOGGER.info(f"{self.label} Executed callback {func!r}")

    def callback(self, func, event_data):
        func = self.resolve_callable(func, event_data)
        if self.send_event:
            func(event_data)
        else:
            func(*event_data.args, **event_data.kwargs)

    def _set_model_states_bindings(self, model):
        for state in self.states.values():
            self._set_model_to_state(state, model)

    def _set_model_to_state(self, state, model):
        model_method_name = f"is_{state.name}"
        func = partial(self.is_state, state.value)
        self._check_model_binding(model, model_method_name, func)
        for state_callback_name in self.state_cls.dynamic_callbacks:
            model_method_name = f"{state_callback_name}_{state.name}"
            if hasattr(model, model_method_name) \
                and inspect.ismethod(getattr(model, model_method_name)) \
                and model_method_name not in getattr(state, state_callback_name):
                state.add_callback(state_callback_name[3:], model_method_name)

    def _set_triggers_to_model(self, model):
        for trigger in self.events:
            self._check_model_binding(model, trigger, partial(self.events[trigger].trigger))

    def _has_state(self, state, raise_error=False):
        found = state in self.states.values()
        if not found and raise_error:
            msg = f"State {state.name if hasattr(state, 'name') else state!r} has not been added to the FSM"
            raise ValueError(msg)
        return found


class FSMMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fsm = self._get_fsm_field()
        self.fsm.init_model(self)

    def _get_fsm_field(self):
        fsm_field_lst = []
        for field in self._meta.fields:
            if isinstance(field, FSMField):
                fsm_field_lst.append(field)

        if not len(fsm_field_lst):
            raise ValueError(f"Not found FSMField on {self!r}")

        if len(fsm_field_lst) > 1:
            raise ValueError(f"Found more than one FSMField, FSMField should be only define one")

        return fsm_field_lst[0]
