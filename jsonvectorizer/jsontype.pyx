import datetime
import sys


# Python version (for handling unicode strings)
cdef int VERSION = sys.version_info.major

cdef JsonType typeof(object doc) except? JNULL:
	# Find the type of the provided JSON document
	if doc is None:
		return JNULL

	json_type = type(doc)
	if json_type is dict:
		return OBJECT
	elif json_type is list:
		return ARRAY
	elif json_type is bool:
		return BOOLEAN
	elif json_type is int:
		return NUMBER
	elif json_type is float:
		return NUMBER
	elif json_type is str:
		try:
			datetime.datetime.strptime(doc, "%Y-%m-%dT%H:%M:%SZ")
			return TIMESTAMP
		except (ValueError,TypeError):
			return STRING
	elif VERSION == 2 and json_type is unicode:
		return STRING
	else:
		raise TypeError('{} is not a valid JSON type'.format(json_type))


cdef JsonType str2type(str json_type) except? JNULL:
	# Convert a string to a JSON type
	json_type = json_type.lower()
	if json_type == 'object':
		return OBJECT
	elif json_type == 'array':
		return ARRAY
	elif json_type == 'null':
		return JNULL
	elif json_type == 'boolean':
		return BOOLEAN
	elif json_type == 'integer':
		return INTEGER
	elif json_type == 'number':
		return NUMBER
	elif json_type == 'timestamp':
		return TIMESTAMP
	elif json_type == 'string':
		return STRING
	else:
		raise ValueError("'{}' is not a valid JSON type".format(json_type))


cdef str type2str(JsonType json_type):
	# Convert a JSON type to a string
	if json_type == OBJECT:
		return 'object'
	elif json_type == ARRAY:
		return 'array'
	elif json_type == JNULL:
		return 'null'
	elif json_type == BOOLEAN:
		return 'boolean'
	elif json_type == INTEGER:
		return 'integer'
	elif json_type == NUMBER:
		return 'number'
	elif json_type == TIMESTAMP:
		return 'timestamp'
	elif json_type == STRING:
		return 'string'
