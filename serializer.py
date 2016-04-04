import sys

from algs import (
    get_class_hierarchy_attrs,
    )


class NOTHING(object):
    pass


class CorruptionError(Exception):
    pass


class SerializerInterface(object):
    """
    To implement SerializerInterface, place this class on a property named
    `_serializer` of your class and provide it with the following attributes.
    The class hierarchy will be walked in method resolution order and the
    attributes compiled in the order they are found.

    `serializable_attrs` is an iterable which is a list of attribute names
    in the order they will be serialized. All attributes named in this
    list will be saved to the stream given to the Serializer in the order
    they are listed. It is mandatory
    `serializers` is a map of attribute names to functions that serialize
    that attribute. It is optional.
    `deserializers` is a map of attribute names to functions that deserialize
    that attribute. It is optional.
    """

    def __init__(self, serializable_attrs,
                 deserializers=None, serializers=None):
        if not serializable_attrs:
            raise ValueError(
                'List of serializable attributes must not be empty')
        self.serializable_attrs = serializable_attrs
        self.deserializers = deserializers
        self.serializers = serializers


class Serializer(object):
    """
    Serializer/Deserializer functions.
    A single object of type SerializerInterface that contains the three
    properties described above must be used and provided in the
    `_serializer` property of a class.
    """

    known_modes = [
        'str',
        'binary', #TODO: EH?
        ]

    def __init__(self, stream=sys.stdout, stream_mode='str'):
        if stream_mode not in self.known_modes:
            raise ValueError('Unknown mode provided to serializer')
        self.stream = stream
        self.stream_mode = stream_mode

    def _get_attr_deserializers(self, klass):
        return get_class_hierarchy_attrs(klass, 'deserializers',
                                         sub_attr='_serializer')

    def _get_attr_serializers(self, klass):
        return get_class_hierarchy_attrs(klass, 'serializers',
                                         sub_attr='_serializer')

    def _get_serializable_attrs(self, klass):
        return get_class_hierarchy_attrs(klass,
                                         'serializable_attrs',
                                         output_type=list,
                                         sub_attr='_serializer')

    def _get_serializable_attrs_and_serializers(self, klass):
        return get_class_hierarchy_attrs(klass,
                                         ['serializable_attrs',
                                          'serializers'],
                                         output_type=list,
                                         sub_attr='_serializer')

    def _get_serializable_attrs_and_deserializers(self, klass):
        return get_class_hierarchy_attrs(klass,
                                         ['serializable_attrs',
                                          'deserializers'],
                                         output_type=list,
                                         sub_attr='_serializer')

    def serialize(self, other, names=None, serializers=None):
        if not names and not serializers:
            attrs, serializers = (
                self._get_serializable_attrs_and_serializers(type(other)))
        elif not names:
            attrs = self._get_serializable_attrs(type(other))
        elif not serializers:
            serializers = self._get_attr_serializers(type(other))

        for k in attrs:
            v = getattr(other, k, NOTHING)
            if v is NOTHING:
                continue
            if k in serializers:
                v = serializers[k](v)

            if self.stream_mode == 'str':
                self.stream.write("{}:{}\n".format(k,v))
            elif self.stream_mode == 'binary':
                self.stream.write("{}".format(k))
                self.stream.write("\n")
                self.stream.write(v)
                self.stream.write("\n")
            else:
                raise ValueError('Unknown mode provided to serializer')

    def deserialize(self, other_klass, names=None, deserializers=None):
        if not names and not deserializers:
            attr_names, deserializers = (
                self._get_serializable_attrs_and_deserializers(other_klass))
        elif not names:
            attr_names = self._get_serializable_attrs(other_klass)
        elif not deserializers:
            deserializers = self._get_attr_deserializers(other_klass)

        attrs = {}
        if self.stream_mode == 'str':
            for line in self.stream.readline().rstrip("\n"):
                k, v = line.split(':', 1)
                if k not in attr_names:
                    raise CorruptionError(
                        'Got {} when valid attributes are {}'.format(
                            k, ','.join(attr_names)))
                if k in deserializers:
                    v = deserializers[k](v)
                attrs[k] = v
        elif self.stream_mode == 'binary':
            pair_idx = 0
            temp = [None, None]
            for line in self.stream.readline().rstrip("\n"):
                temp[pair_idx] = line
                pair_idx = (pair_idx + 1) % 2
                if pair_idx == 0:
                    k = temp[0]
                    if k not in attr_names:
                        raise CorruptionError(
                            'Got {} when valid attributes are {}'.format(
                                k, ','.join(attr_names)))
                    v = temp[1]
                    if k in deserializers:
                        v = deserializers[k](v)
                    attrs[k] = v
        else:
            raise ValueError('Unknown mode provided to serializer')
        return other_klass(**attrs)
