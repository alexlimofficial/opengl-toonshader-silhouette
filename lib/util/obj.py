import numpy as np


class ObjLoader:
    """ObjLoader class which reads from a .obj file and loads the data."""
    @staticmethod
    def extract_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == 'float':
                coordinates.append(float(d))
            elif data_type == 'int':
                coordinates.append(int(d)-1)
            else:
                raise Exception(f'data_type must be float or int')

    @staticmethod
    def load_model(file, sorted=True):
        vert_coords = []    # all the vertex coordinates
        tex_coords = []     # all the texture coordinates
        norm_coords = []    # all the vertex normals

        vert_indices = []   # all the vertex indices
        tex_indices = []    # all the texture indices
        norm_indices = []   # all the normal indices

        expanded_face_indices = []

        with open(file, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                # found a vertex [v, x, y, z]
                if values[0] == 'v':
                    ObjLoader.extract_data(values, vert_coords, 'v', 'float')
                # found a texture coord [vt, u, v]
                elif values[0] == 'vt':
                    ObjLoader.extract_data(values, tex_coords, 'vt', 'float')
                # found a vertex normal [vn, x, y, z]
                elif values[0] == 'vn':
                    ObjLoader.extract_data(values, norm_coords, 'vn', 'float')
                # found a face
                elif values[0] == 'f':
                    for value in values[1:]:
                        val = value.split('/')

                        # subtract 1 as .obj is 1-indexed but we want 0-indexed
                        v1 = int(val[0]) - 1
                        vert_indices.append(v1)
                        expanded_face_indices.append(v1)

                        v2 = int(val[1]) - 1
                        tex_indices.append(v2)
                        expanded_face_indices.append(v2)

                        v3 = int(val[2]) - 1
                        norm_indices.append(v3)
                        expanded_face_indices.append(v3)
                else:
                    pass

                line = f.readline()

        buffer = []
        for i, ind in enumerate(expanded_face_indices):
            if i % 3 == 0:
                start = ind * 3
                end = start + 3
                buffer.extend(vert_coords[start:end])
            elif i % 3 == 1:
                start = ind * 2
                end = start + 2
                buffer.extend(tex_coords[start:end])
            elif i % 3 == 2:
                start = ind * 3
                end = start + 3
                buffer.extend(norm_coords[start:end])

        reshape_expanded_indices = np.asarray(expanded_face_indices).reshape(-1, 3, 3)
        assert np.allclose(reshape_expanded_indices[:, :, 0].flatten(), np.asarray(vert_indices))
        assert np.allclose(reshape_expanded_indices[:, :, 1].flatten(), np.asarray(tex_indices))
        assert np.allclose(reshape_expanded_indices[:, :, 2].flatten(), np.asarray(norm_indices))

        meta = {
            'v': np.asarray(vert_coords, dtype='float32'),
            'vt': np.asarray(tex_coords, dtype='float32'),
            'vn': np.asarray(norm_coords, dtype='float32'),
            'vert_indices': np.asarray(vert_indices, dtype='uint32'),
            'tex_indices': np.asarray(tex_indices, dtype='uint32'),
            'norm_indices': np.asarray(norm_indices, dtype='uint32'),
            'expanded_face_indices': np.asarray(expanded_face_indices, dtype='uint32'),
            'indices': np.arange(reshape_expanded_indices.shape[0]*reshape_expanded_indices.shape[1]),
            'buffer': np.asarray(buffer, dtype='float32')
        }

        return meta
