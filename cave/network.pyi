from typing import List, Dict, Tuple, Any, Generic, TypeVar
import cave


# The Package is part of cave's Network module and it's meant to provide Packing capabilities for you to transform regular data types such as strings, integers, floats and vectors into a Packed format that can be transmitted over network.
class PackageException():
	"""
	Exception class raised by the Package class when something unexpected or wrong happens during its operations.
	"""

	msg: str # Variable containing the Exception message.
	def __init__(self, _msg: str): ... 
	def what(self) -> str: ... # Returns the Exception message. Same as the "msg" variable.


class Package():
	"""
	Package class that allows you to pack regular data into a network friendly binary format by writting the data to it and/or unpack this same data by reading from it.

If you need to know more about the internal format, here is the explanation: For each data, it first writes the data type as a `char`, then the data itself. The package does not have any header or extra data other than the one you write to it.
	"""


	# Enum: networkPackagePackageMode:# "Package" class Enumeration that specifies if the Package is set to Read or Write mode. Only the corresponding read/write methods will work.
	READ : int # Enables reading data back from the Package to a regular format.
	WRITE : int # Enables writting regular data into the Package.


	# Enum: networkPackageDataType:# "Package" class Enumeration, specifies the supported data types. This is usually handled by you automatically by the Package methods.
	TYPE_INVALID : int 
	TYPE_INT : int 
	TYPE_LONG_INT : int # Represents a size_t.
	TYPE_FLOAT : int 
	TYPE_BOOL : int 
	TYPE_VEC3 : int 
	TYPE_STRING : int 

	def __init__(self, readMode: bool = False): ... # Default Package constructor. It sets the Package mode to WRITE by default, unless you pass True as a parameter (for readMode), then it will be set to READ.
	def __init__(self, data: None, len: int): ... 
	def __init__(self, other: network.Package): ... 
	def writeInt(self, value: int) -> None: ... # Writes a 32 bit integer into the Package. Note that python integers may be higher than 32 bits and it will cause an error.
	def writeLongInt(self, value: int) -> None: ... # Writes a 64 bit unsigned integer (size_t) to the Package.
	def writeFloat(self, value: float) -> None: ... 
	def writeBool(self, value: bool) -> None: ... 
	def writeVector3(self, value: cave.Vector3) -> None: ... 
	def writeString(self, value: str) -> None: ... # Writes a regular string to the Package. If it's important for you to know it internally, it will write the TYPE_STRING, followed by the string length N and then N char representing the string itself.
	def readInt(self) -> int: ... # Reads a 32 bit integer from the Package.
	def readLongInt(self) -> int: ... # Reads a 64 bit unsigned integer (size_t) from the Package.
	def readFloat(self) -> float: ... 
	def readBool(self) -> bool: ... 
	def readVector3(self) -> cave.Vector3: ... 
	def readString(self) -> str: ... # Reads a regular string from the Package.
	def getNextDataType(self) -> network.Package.DataType: ... # If the mode is set to READ, returns the next data type about to be read. Returns TYPE_INVALID, otherwise.
	def getBufferSize(self) -> int: ... # Returns the size of the current Read/Write buffer of the Package.



# Part of Cave's Network system, allows you to quickly create a Server for Online (Multiplayer) games.
# Cave's Networking is built on top of ENET, which  provides a relatively thin, simple and robust network communication layer on top of UDP. 
class ServerPeer():
	"""
	When a Client connects to a Cave Server, it is considered by the server as a Peer and this is the class that handles it.
	"""

	def __init__(self, peer: None = None): ... 
	def __init__(self, other: network.ServerPeer): ... 
	def send(self, pkg: network.Package, reliable: bool = True) -> None: ... # Sends the given Package to this Peer alone, allowing the Server to directly communicate to a single Client. If reliable is set to True, it will guarantee that the Package arrives to the server. If not, it will simply send it once and that's it. Please notice that, due to how internet works, reliable packages are considerably slower.
	def getAddress(self) -> str: ... # Returns the Peer's Address, usually represented by its ip + port, e.g.: "123.123.1.7:33333".
	def getHost(self) -> str: ... # Returns the Peer's host name (IP) as a string.
	def getPort(self) -> int: ... # Returns the Peer's port number.


class ServerPackage():
	"""
	When receiving Packages in the Server, since they can arrive from multiple different Peers, they get wrapped around this ServerPackage class to properly identify its sender (origin).
	"""

	sender: network.ServerPeer # The ServerPeer who've sent this Package.
	package: network.Package # The Package itself.
	def __init__(self): ... 
	def __init__(self, sndr: network.ServerPeer, pkg: network.Package): ... 
	def __init__(self, other: network.ServerPackage): ... 


class Server():
	"""
	The main Cave's Network Server class that allows you to open a Server that Cave (or Enet) Clients can connect to, sending and receiving, reliable or unreliable, messages over network.
	"""

	def __init__(self, addr: str = "localhost", port: int = 33333): ... # Creates the Server and binds it to the provided address and port number. You can use "localhost" (default) address to make the Server visible to the machine currently running it only or "0.0.0.0" to make it visible to the entire network.
	def update(self) -> None: ... # For the Server to function properly, you **MUST** call this once at the end of every frame.
	def sendToAll(self, pkg: network.Package, reliable: bool = True) -> None: ... # Given a Package, sends it to ALL connected Peers (Clients). As of in all other sending methods, you can also choose if you want this to be reliable or not.
	def popPackages(self) -> List[network.ServerPackage]: ... # Returns a list of all cave.network.ServerPackage received from the Clients (Peers) since the last time you called this method.
	def getPeers(self) -> List[network.ServerPeer]: ... # Returns a list of all connected cave.network.ServerPeer (Clients).
	def isValid(self) -> bool: ... # Returns True if this Server is valid. If False, something wrong happened when creating it and it won't function at all.
	def getNumClients(self) -> int: ... 





# Part of Cave's Network system, allows you to quickly create a Client for Online (Multiplayer) games.
# Cave's Networking is built on top of ENET, which  provides a relatively thin, simple and robust network communication layer on top of UDP. 
class Client():
	"""
	Client class that allows you to connect to a Cave or Enet Server, sending and receiving, reliable or unreliable, messages over network.
	"""

	def __init__(self, addr: str = "localhost", port: int = 33333): ... # Creates the client and starts the attempt to connect to the provided Server (by the address and port).
	def isValid(self) -> bool: ... # Returns True if the Client host is valid. If it's false, it means that there was some internal network error that caused it to fail to create the host and/or sockets.
	def isConnected(self) -> bool: ... # Returns True if the Client successfully managed to connect to the provided server. A valid Client may still return False here while it is attempting to connect, since this relies on the internet connection and many other factors to successfully connect to the Server. You should handle this "waiting to Connect..." time in your Game code.
	def update(self) -> None: ... # For the Client to function properly, you **MUST** call this once at the end of every frame.
	def send(self, pkg: network.Package, reliable: bool = True) -> None: ... # Given a Package, sends it to the server. If reliable is set to True, it will guarantee that the Package arrives to the server. If not, it will simply send it once and that's it. Please notice that, due to how internet works, reliable packages are considerably slower.
	def popPackages(self) -> List[network.Package]: ... # Returns a list of all cave.network.Package received from the Server since the last time you called this method.


