function guard = trainingLock(folder)
% The OS releases the exclusive handle even if MATLAB crashes.
path=fullfile(folder,'training.lock');
try
    if ispc
        handle=System.IO.File.Open(path,System.IO.FileMode.OpenOrCreate, ...
            System.IO.FileAccess.ReadWrite,System.IO.FileShare.None);
        guard=onCleanup(@() handle.Dispose());
    else
        assert(usejava('jvm'),'S8:Lock','Training requires JVM file locking on this platform.');
        handle=javaObject('java.io.RandomAccessFile',path,'rw');
        channel=handle.getChannel();
        lock=channel.tryLock();
        assert(~isempty(lock),'S8:Busy','Training folder is already locked.');
        guard=onCleanup(@() release(lock,channel,handle));
    end
catch cause
    error('S8:Busy','Cannot acquire exclusive training lock for %s. Another trainer may be running. %s',folder,cause.message);
end
end

function release(lock,channel,handle)
lock.release(); channel.close(); handle.close();
end
