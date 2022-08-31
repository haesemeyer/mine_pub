#  Copyright (c) 2019. Martin Haesemeyer. All rights reserved.
#
#  Licensced under the MIT license. See LICENSE

"""
Class wrapper of experiments with support of serialization to/from HDF5 files
"""
import warnings

import h5py
import numpy as np
from datetime import datetime
from os import path
import json
from typing import Dict, List, Optional, Any, Union


class ExperimentException(Exception):
    """
    Exception to signal that invalid operation was performed on Experiment
    """
    def __init__(self, message: str):
        super().__init__(message)


class LazyLoadObject:
    """Class to represent a lazy-loaded object including current status"""

    def __init__(self, backend: Optional[h5py.File], storage_name: str, lazy_loaded=True):
        """
        Creates a new LazyLoadObject
        :param backend: The hdf5 backend store for this data
        :param storage_name: The key of this data in the hdf5 file
        :param lazy_loaded: Whether the object is lazy loaded or set outright
        """
        self.__my_backend: Optional[h5py.File] = backend
        self.__my_name: str = storage_name
        self.__am_i_lazy: bool = lazy_loaded
        self.__my_data: Optional[List[np.ndarray]] = None
        self.__am_i_modified: bool = False

    def get_data(self, n_planes) -> List[np.ndarray]:
        if self.__am_i_lazy and self.__my_backend is None:
            raise ExperimentException("Object is for lazy load but no backend specified")
        if self.__am_i_lazy:
            assert self.__my_data is None
            self.__my_data = []
            for i in range(n_planes):
                plane_group = self.__my_backend[str(i)]
                if self.__my_name in plane_group:  # test if this experiment had the requested data
                    self.__my_data.append(plane_group[self.__my_name][()])
            self.__am_i_lazy = False
        return self.__my_data

    def set_data(self, value: List[np.ndarray]) -> None:
        self.__am_i_lazy = False
        self.__am_i_modified = True
        self.__my_data = value

    def update(self) -> None:
        if self.am_i_modified and len(self.__my_data) > 0:
            for i in range(len(self.__my_data)):
                plane_group = self.__my_backend[str(i)]
                if self.__my_name in plane_group:
                    del plane_group[self.__my_name]
                plane_group.create_dataset(self.__my_name, data=self.__my_data[i], compression="gzip",
                                           compression_opts=5)

    @property
    def am_i_lazy(self):
        return self.__am_i_lazy

    @property
    def am_i_modified(self):
        return self.__am_i_modified


class Experiment2P:
    """
    Represents a 2-photon imaging experiment on which cells have been segmented
    """
    def __init__(self, lazy_load_filename: str = "", allow_write=True):
        """
        Creates a new Experiment2P object
        :param lazy_load_filename: If given, will attach this instance to an existing hdf5 store at the filename
        :param allow_write: If lazy load filename is given and set to true, allow changing of store
        """
        self.info_data: Dict[str, Any] = {}  # data from the experiment's info data
        self.experiment_name: str = ""  # name of the experiment
        self.original_path: str = ""  # the original path when the experiment was analyzed
        self.scope_name: str = ""  # the name assigned to the microscope for informational purposes
        self.comment: str = ""  # general comment associated with the experiment
        self.tail_frame_rate: int = 0  # the frame-rate of the tail camera
        self.tail_data_augmented: bool = False
        self.scanner_data: List[Dict] = []  # for each experimental plane the associated scanner data (resolution, etc.)
        self.bout_data: List[np.ndarray] = []  # for each experimental plane, matrix of extracted swim bouts
        self.tail_frame_times: List[np.ndarray] = []  # for each plane, the scan relative time of each tail cam frame

        # the following are optionally lazy-loaded data objects #
        # for each experimental plane the tail data if applicable
        self.__tail_data: LazyLoadObject = LazyLoadObject(None, "tail_data", False)
        # if tail data is augmented for each plane replaced frames
        self.__replaced_tail_frames: LazyLoadObject = LazyLoadObject(None, "replaced_tail_frames", False)
        # per plane 20Hz vector of laser command voltages if applicable
        self.__laser_data: LazyLoadObject = LazyLoadObject(None, "laser_data", False)
        # for each experimental plane the inferred calcium of each extracted unit
        self.__all_c: LazyLoadObject = LazyLoadObject(None, "C", False)
        # for each experimental plane the dF/F of each extracted unit
        self.__all_dff: LazyLoadObject = LazyLoadObject(None, "dff", False)
        # for each plane the realigned 8-bit functional stack
        self.__func_stacks: LazyLoadObject = LazyLoadObject(None, "func_stack", False)
        # end optionally lazy-loaded data objects #

        # for each experimental plane the unit centroid coordinates as (x [col]/y [row]) pairs
        self.all_centroids: List[np.ndarray] = []
        # for each experimental plane the size of each unit in pixels (not weighted)
        self.all_sizes: List[np.ndarray] = []
        # for each experimental plane n_comp x 4 array <component-ix, weight, x-coord, y-coord>
        self.all_spatial: List[np.ndarray] = []
        # list of 32 bit plane projections after motion correction
        self.projections: List[np.ndarray] = []
        # for dual-channel experiments, list of 32 bit plane projections of anatomy channel
        self.anat_projections: List[np.ndarray] = []
        self.mcorr_dicts: List[Dict] = []  # the motion correction parameters used on each plane
        self.cnmf_extract_dicts: List[Dict] = []  # the cnmf source extraction parameters used on each plane
        self.cnmf_val_dicts: List[Dict] = []  # the cnmf validation parameters used on each plane
        self.version: str = "2"  # version ID for future-proofing
        self.populated: bool = False  # indicates if class contains experimental data through analysis or loading
        self.lazy: bool = False  # indicates that we have lazy-loaded and attached to hdf5 file
        self.__hdf5_store: Optional[h5py.File] = None
        if lazy_load_filename != "" and not path.exists(lazy_load_filename):
            raise ValueError(f"The file {lazy_load_filename} does not exist. Cannot attach to store.")
        if lazy_load_filename != "":
            self.__lazy_load(lazy_load_filename, allow_write)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__hdf5_store is not None:
            self.__hdf5_store.close()
            self.__hdf5_store = None
            self.lazy = False

    def __lazy_load(self, lazy_load_filename: str, allow_write: bool) -> None:
        """
        Performs "lazy loading" loading some properties and making others available through an hdf5 store backend
        :param lazy_load_filename: The name of the experiment hdf5 file
        :param allow_write: If set to true, file will be opened in append mode
        """
        self.__hdf5_store = h5py.File(lazy_load_filename, 'r+' if allow_write else 'r')
        self.version = self.__hdf5_store["version"][()]
        try:
            if self.version == b"unstable" or self.version == "unstable":
                warnings.warn("Experiment file was created with development version of analysis code. Trying to "
                              "load as version 1")
                self.version = "0"
            elif int(self.version) > 2:
                self.__hdf5_store.close()
                self.__hdf5_store = None
                raise IOError(f"File version {self.version} is larger than highest recognized version '2'")
        except ValueError:
            self.__hdf5_store.close()
            self.__hdf5_store = None
            raise IOError(f"File version {self.version} not recognized")
        # load general experiment data
        n_planes = self.__hdf5_store["n_planes"][()]
        self.experiment_name = self.__hdf5_store["experiment_name"][()]
        self.original_path = self.__hdf5_store["original_path"][()]
        self.scope_name = self.__hdf5_store["scope_name"][()]
        self.comment = self.__hdf5_store["comment"][()]
        self.tail_frame_rate = self.__hdf5_store["tail_frame_rate"][()]
        # load singular parameter dictionary
        self.info_data = self._load_dictionary("info_data", self.__hdf5_store)
        # load tail-data modification flag if this is version 2
        if int(self.version) > 1:
            self.tail_data_augmented = self.__hdf5_store["tail_data_augmented"][()]
        # load some per-plane data leave larger objects unloaded
        for i in range(n_planes):
            plane_group = self.__hdf5_store[str(i)]
            self.scanner_data.append(self._load_dictionary("scanner_data", plane_group))
            self.projections.append(plane_group["projection"][()])
            if "anat_projection" in plane_group:  # test if this experiment was dual-channel
                self.anat_projections.append(plane_group["anat_projection"][()])
            if "tail_data" in plane_group:  # test if this experiment had tail data (for all planes)
                self.bout_data.append(plane_group["bout_data"][()])
                self.tail_frame_times.append(plane_group["tail_frame_time"][()])
            self.all_centroids.append(plane_group["centroids"][()])
            self.all_sizes.append(plane_group["sizes"][()])
            self.all_spatial.append(plane_group["spatial"][()])
            ps = plane_group["mcorr_dict"][()]
            self.mcorr_dicts.append(json.loads(ps))
            ps = plane_group["cnmf_extract_dict"][()]
            self.cnmf_extract_dicts.append(json.loads(ps))
            ps = plane_group["cnmf_val_dict"][()]
            self.cnmf_val_dicts.append(json.loads(ps))
        # update lazy load stores for non-loaded objects
        self.__tail_data = LazyLoadObject(self.__hdf5_store, "tail_data")
        self.__replaced_tail_frames = LazyLoadObject(self.__hdf5_store, "replaced_tail_frames")
        self.__laser_data = LazyLoadObject(self.__hdf5_store, "laser_data")
        self.__all_c = LazyLoadObject(self.__hdf5_store, "C")
        self.__all_dff = LazyLoadObject(self.__hdf5_store, "dff")
        self.__func_stacks = LazyLoadObject(self.__hdf5_store, "func_stack")
        self.lazy = True

    @property
    def tail_data(self) -> List[np.ndarray]:
        return self.__tail_data.get_data(self.n_planes)

    @tail_data.setter
    def tail_data(self, value: List[np.ndarray]):
        self.__tail_data.set_data(value)

    @property
    def replaced_tail_frames(self) -> List[np.ndarray]:
        return self.__replaced_tail_frames.get_data(self.n_planes)

    @replaced_tail_frames.setter
    def replaced_tail_frames(self, value: List[np.ndarray]):
        self.__replaced_tail_frames.set_data(value)

    @property
    def laser_data(self) -> List[np.ndarray]:
        return self.__laser_data.get_data(self.n_planes)

    @laser_data.setter
    def laser_data(self, value: List[np.ndarray]):
        self.__laser_data.set_data(value)

    @property
    def all_c(self) -> List[np.ndarray]:
        return self.__all_c.get_data(self.n_planes)

    @all_c.setter
    def all_c(self, value: List[np.ndarray]):
        self.__all_c.set_data(value)

    @property
    def all_dff(self) -> List[np.ndarray]:
        return self.__all_dff.get_data(self.n_planes)

    @all_dff.setter
    def all_dff(self, value: List[np.ndarray]):
        self.__all_dff.set_data(value)

    @property
    def func_stacks(self) -> List[np.ndarray]:
        return self.__func_stacks.get_data(self.n_planes)

    @func_stacks.setter
    def func_stacks(self, value: List[np.ndarray]):
        self.__func_stacks.set_data(value)

    @staticmethod
    def load_experiment(file_name: str):
        """
        Loads an experiment from a serialization in an hdf5 file
        :param file_name: The name of the hdf5 file storing the experiment
        :return: Experiment object with all relevant data
        """
        exp = Experiment2P()
        # initialize the lazy-load objects with empty lists
        exp.tail_data = []
        exp.replaced_tail_frames = []
        exp.laser_data = []
        exp.all_c = []
        exp.all_dff = []
        exp.func_stacks = []
        with h5py.File(file_name, 'r') as dfile:
            exp.version = dfile["version"][()]  # in future allows for version specific loading
            try:
                if exp.version == b"unstable" or exp.version == "unstable":
                    warnings.warn("Experiment file was created with development version of analysis code. Trying to "
                                  "load as version 1")
                elif int(exp.version) > 2:
                    raise IOError(f"File version {exp.version} is larger than highest recognized version '2'")
            except ValueError:
                raise IOError(f"File version {exp.version} not recognized")
            # load general experiment data
            n_planes = dfile["n_planes"][()]  # inferrred property of class but used here for loading plane data
            exp.experiment_name = dfile["experiment_name"][()]
            exp.original_path = dfile["original_path"][()]
            exp.scope_name = dfile["scope_name"][()]
            exp.comment = dfile["comment"][()]
            exp.tail_frame_rate = dfile["tail_frame_rate"][()]
            # load singular parameter dictionary
            exp.info_data = exp._load_dictionary("info_data", dfile)
            # load tail-data modification flag if this is version 2
            if int(exp.version) > 1:
                exp.tail_data_augmented = dfile["tail_data_augmented"][()]
            # load per-plane data
            for i in range(n_planes):
                plane_group = dfile[str(i)]
                exp.scanner_data.append(exp._load_dictionary("scanner_data", plane_group))
                exp.tail_data.append(plane_group["tail_data"][()])
                exp.projections.append(plane_group["projection"][()])
                if "func_stack" in plane_group:
                    exp.func_stacks.append(plane_group["func_stack"][()])
                if "anat_projection" in plane_group:  # test if this experiment was dual-channel
                    exp.anat_projections.append(plane_group["anat_projection"][()])
                if "tail_data" in plane_group:  # test if this experiment had tail data (for all planes)
                    exp.tail_data.append(plane_group["tail_data"][()])
                    exp.bout_data.append(plane_group["bout_data"][()])
                    exp.tail_frame_times.append(plane_group["tail_frame_time"][()])
                if int(exp.version) > 1 and "replaced_tail_frames" in plane_group:
                    exp.replaced_tail_frames.append(plane_group["replaced_tail_frames"][()])
                if "laser_data" in plane_group:  # test if this experiment had laser data
                    exp.laser_data.append(plane_group["laser_data"][()])
                exp.all_c.append(plane_group["C"][()])
                exp.all_dff.append(plane_group["dff"][()])
                exp.all_centroids.append(plane_group["centroids"][()])
                exp.all_sizes.append(plane_group["sizes"][()])
                exp.all_spatial.append(plane_group["spatial"][()])
                ps = plane_group["mcorr_dict"][()]
                exp.mcorr_dicts.append(json.loads(ps))
                ps = plane_group["cnmf_extract_dict"][()]
                exp.cnmf_extract_dicts.append(json.loads(ps))
                ps = plane_group["cnmf_val_dict"][()]
                exp.cnmf_val_dicts.append(json.loads(ps))
        exp.populated = True
        return exp

    @staticmethod
    def _save_dictionary(d: dict, dict_name: str, file: h5py.File) -> None:
        """
        Saves a dictionary to hdf5 file. Note: Does not work for general dictionaries!
        :param d: The dictionary to save
        :param dict_name: The name of the dictionary
        :param file: The hdf5 file to which the dictionary will be added
        """
        g = file.create_group(dict_name)
        for k in d:
            if "finish_time" in k or "start_time" in k:
                # need to encode datetime object as string
                date_time_string = d[k].strftime("%m/%d/%Y %I:%M:%S %p")
                g.create_dataset(k, data=date_time_string)
            else:
                g.create_dataset(k, data=d[k])

    @staticmethod
    def _load_dictionary(dict_name: str, file: h5py.File) -> Dict:
        """
        Loads a experiment related dictionary from file
        :param dict_name: The name of the dictionary
        :param file: The hdf5 file containing the dictionary
        :return: The populated dictionary
        """
        d = {}
        g = file[dict_name]
        for k in g:
            if "finish_time" in k or "start_time" in k:
                # need to decode byte-string into datetime object
                try:
                    date_time_string = g[k][()].decode('UTF-8')  # convert bye string to string
                except AttributeError:
                    # it is already a unicode string
                    date_time_string = g[k][()]
                d[k] = datetime.strptime(date_time_string, "%m/%d/%Y %I:%M:%S %p")
            else:
                d[k] = g[k][()]
        return d

    @staticmethod
    def _update_data(storage: Union[h5py.File, h5py.Group], name: str, data: Any, compress: bool = False) -> None:
        """
        Updates data in an hdf5 file or group deleting old data if necessary
        :param storage: The hdf5 file or group that should store the new data
        :param name: The key name of the new data
        :param data: The data
        :param compress: If true, data will be stored compressed
        """
        if name in storage:
            del storage[name]
        if compress:
            storage.create_dataset(name, data=data, compression="gzip", compression_opts=5)
        else:
            storage.create_dataset(name, data=data)

    def update_experiment_store(self) -> None:
        if not self.lazy:
            raise ExperimentException("Only lazy-loaded experiments can be updated. Use save_experiment otherwise")
        # data that isn't lazy-loaded is usually small and will be updated by default. Lazy-loaded objects will only
        # be written to disk if they 1) were loaded and 2) were modified
        # no dictionaries will be modified, similarly basic experiment info data will not be updated
        self.version = "2"
        self._update_data(self.__hdf5_store, "version", self.version)
        self._update_data(self.__hdf5_store, "n_planes", data=self.n_planes)
        self._update_data(self.__hdf5_store, "tail_frame_rate", data=self.tail_frame_rate)
        # update all lazy-loaded data if indicated
        self.__tail_data.update()
        if int(self.version) > 1:
            self._update_data(self.__hdf5_store, "tail_data_augmented", data=self.tail_data_augmented)
            self.__replaced_tail_frames.update()
        self.__laser_data.update()
        self.__all_c.update()
        self.__all_dff.update()
        self.__func_stacks.update()
        # save per-plane non-lazy data
        for i in range(self.n_planes):
            plane_group = self.__hdf5_store[str(i)]
            if len(self.tail_data) > 0:
                if self.bout_data[i] is not None:
                    self._update_data(plane_group, "bout_data", self.bout_data[i], True)
                else:
                    # no bouts were found, save dummy array of one line of np.nan
                    bd = np.full((1, 8), np.nan)
                    self._update_data(plane_group, "bout_data", bd, True)
                self._update_data(plane_group, "tail_frame_time", self.tail_frame_times[i], True)
            self._update_data(plane_group, "projection", self.projections[i], True)
            if len(self.anat_projections) > 0:  # this is a dual-channel experiment
                self._update_data(plane_group, "anat_projection", self.anat_projections[i], True)
            self._update_data(plane_group, "centroids", self.all_centroids[i], True)
            self._update_data(plane_group, "sizes", self.all_sizes[i], True)
            self._update_data(plane_group, "spatial", self.all_spatial[i], True)

    def save_experiment(self, file_name: str, ovr_if_exists=False) -> None:
        """
        Saves the experiment to the indicated file in hdf5 format
        :param file_name: The name of the file to save to
        :param ovr_if_exists: If set to true and file exists it will be overwritten otherwise exception will be raised
        """
        if not self.populated:
            raise ExperimentException("Empty experiment class cannot be saved. Load or analyze experiment first.")
        if ovr_if_exists:
            dfile = h5py.File(file_name, "w")
        else:
            dfile = h5py.File(file_name, "x")
        try:
            dfile.create_dataset("version", data=self.version)  # for later backwards compatibility
            # save general experiment data
            dfile.create_dataset("experiment_name", data=self.experiment_name)
            dfile.create_dataset("original_path", data=self.original_path)
            dfile.create_dataset("scope_name", data=self.scope_name)
            dfile.create_dataset("comment", data=self.comment)
            dfile.create_dataset("n_planes", data=self.n_planes)
            dfile.create_dataset("tail_frame_rate", data=self.tail_frame_rate)
            # save singular parameter dictionary
            self._save_dictionary(self.info_data, "info_data", dfile)
            # save augmentation flag
            if int(self.version) > 1:
                dfile.create_dataset("tail_data_augmented", data=self.tail_data_augmented)
            # save per-plane data
            for i in range(self.n_planes):
                plane_group = dfile.create_group(str(i))
                self._save_dictionary(self.scanner_data[i], "scanner_data", plane_group)
                if len(self.tail_data) > 0:
                    plane_group.create_dataset("tail_data", data=self.tail_data[i], compression="gzip",
                                               compression_opts=5)
                    if self.bout_data[i] is not None:
                        plane_group.create_dataset("bout_data", data=self.bout_data[i], compression="gzip",
                                                   compression_opts=5)
                    else:
                        # no bouts were found, save dummy array of one line of np.nan
                        bd = np.full((1, 8), np.nan)
                        plane_group.create_dataset("bout_data", data=bd, compression="gzip", compression_opts=5)
                    plane_group.create_dataset("tail_frame_time", data=self.tail_frame_times[i])
                if int(self.version) > 1 and len(self.replaced_tail_frames) > 0:
                    plane_group.create_dataset("replaced_tail_frames", data=self.replaced_tail_frames[i],
                                               compression="gzip", compression_opts=5)
                if len(self.laser_data) > 0:
                    plane_group.create_dataset("laser_data", data=self.laser_data[i], compression="gzip",
                                               compression_opts=5)
                plane_group.create_dataset("projection", data=self.projections[i], compression="gzip",
                                           compression_opts=5)
                plane_group.create_dataset("func_stack", data=self.func_stacks[i], compression="gzip",
                                           compression_opts=5)
                if len(self.anat_projections) > 0:  # this is a dual-channel experiment
                    plane_group.create_dataset("anat_projection", data=self.anat_projections[i], compression="gzip",
                                               compression_opts=5)
                plane_group.create_dataset("C", data=self.all_c[i], compression="gzip", compression_opts=5)
                plane_group.create_dataset("dff", data=self.all_dff[i], compression="gzip", compression_opts=5)
                plane_group.create_dataset("centroids", data=self.all_centroids[i], compression="gzip",
                                           compression_opts=5)
                plane_group.create_dataset("sizes", data=self.all_sizes[i], compression="gzip", compression_opts=5)
                plane_group.create_dataset("spatial", data=self.all_spatial[i], compression="gzip", compression_opts=5)
                # due to mixed python types in caiman parameter dictionaries these currently get pickled
                ps = json.dumps(self.mcorr_dicts[i])
                plane_group.create_dataset("mcorr_dict", data=ps)
                ps = json.dumps(self.cnmf_extract_dicts[i])
                plane_group.create_dataset("cnmf_extract_dict", data=ps)
                ps = json.dumps(self.cnmf_val_dicts[i])
                plane_group.create_dataset("cnmf_val_dict", data=ps)
        finally:
            dfile.close()

    def avg_component_brightness(self, use_anat: bool) -> List[np.ndarray]:
        """
        Computes the brightness of each identified component on the functional or anatomical channel
        :param use_anat: If True returns the average brightness of each component on anatomy not functional channel
        :return: n_planes long list of vectors with the time-average brightness of each identified component
        """
        if not self.populated and not self.lazy:
            raise ExperimentException("Experiment does not have data. Use Analyze or Load first.")
        if use_anat and not self.is_dual_channel:
            raise ValueError("Experiment does not have anatomy channel")
        p = self.anat_projections if use_anat else self.projections
        acb = []
        for i in range(self.n_planes):
            # <component-ix, weight, x-coord, y-coord>
            n_components = int(np.max(self.all_spatial[i][:, 0]) + 1)  # component indices are 0-based
            br = np.zeros(n_components, dtype=np.float32)
            for j in range(n_components):
                this_component = self.all_spatial[i][:, 0].astype(int) == j
                spatial_x = self.all_spatial[i][this_component, 2].astype(int)
                spatial_y = self.all_spatial[i][this_component, 3].astype(int)
                br[j] = np.mean(p[i][spatial_y, spatial_x])
            acb.append(br)
        return acb

    @property
    def n_planes(self) -> int:
        return len(self.scanner_data)

    @property
    def is_dual_channel(self) -> bool:
        return len(self.anat_projections) > 0
