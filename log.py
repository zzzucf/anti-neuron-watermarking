import time

class Log(object):
	def __init__(self, filename):
		super(Log, self).__init__()

		self.filename = filename

		# Clear the content if any.
		open(filename, 'w').close()
		
	def _log(self, t, msg):
		logStr = "{} {} | {}".format(t, time.ctime(), msg)
		print(logStr)

		with open(self.filename, 'a') as file:
			file.write(logStr + '\n')

	def info(self, msg):
		self._log("Info", str(msg))

	def error(self, msg):
		self._log("Error", str(msg))

	def warning(self, msg):
		self._log("Warning", str(msg))